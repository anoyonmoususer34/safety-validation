# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Turn a forwarded LiDAR spin into obstacles for the safety validator.

Runs the ``physical_ai_av_lidar`` pipeline (preprocess -> DBSCAN detection) on
the points expressed around the sensor but levelled to the rig orientation,
tracks the detections across spins, then moves the confirmed ones into the rig
frame (x forward, y left), the same frame as the model's candidate waypoints.

A rendered spin yields hundreds of clusters, so a cluster counts only once it has
recurred.  Its velocity relative to the road, with the ego's own driving and
turning taken out, is smoothed over the spins of its track by an alpha-beta
filter that also estimates its acceleration, so a braking vehicle's speed does
not lag behind.

Output obstacle format (consumed by ``SafetyValidator``)::

    {"id", "x", "y", "vx", "vy", "half_length_m", "half_width_m", "confidence",
     "source": "lidar"}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from alpasim_utils.lidar_payload import LidarSpin
from physical_ai_av_lidar.detection import (
    DetectionConfig,
    LidarDetector,
)
from physical_ai_av_lidar.loader import SpinFrame
from physical_ai_av_lidar.preprocessing import LidarPreprocessor, PreprocessConfig
from physical_ai_av_lidar.schema import LidarExtrinsics

logger = logging.getLogger(__name__)

_CONFIRM_SPINS = 3  # spins a cluster must be seen in before it is an obstacle
# How far a cluster may be from where its track predicts.  A new track assumes
# a stationary object, so its first match allows for a moving one.
_NEW_TRACK_GATE_M = 3.0
_TRACK_GATE_M = 1.5
# Alpha-beta filter on a track's velocity: weights of the newest velocity
# measurement's residual in the velocity and (per second) in the acceleration.
_SPEED_GAIN = 0.3
_ACCEL_GAIN = 0.1
_MAX_ACCEL_MPS2 = 10.0  # |acceleration| a road vehicle can reach


@dataclass(frozen=True)
class _Track:
    xy: np.ndarray  # (2,) rig frame, moving with the ego
    velocity: np.ndarray  # (2,) m/s relative to the road, in the rig frame
    accel: np.ndarray  # (2,) m/s^2
    spins: int


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _update_tracks(
    tracks: list[_Track],
    centers: np.ndarray,
    ego_speed: float,
    ego_yaw_rate: float,
    dt_s: float,
) -> list[_Track]:
    """Associate (N, 2) rig-frame cluster centres with ``tracks``, one new track per
    centre.  Between spins the rig drives ``ego_speed * dt_s`` forward and turns by
    ``ego_yaw_rate * dt_s``; both are taken out of the measured velocities."""
    # Previous rig frame -> this one: drive forward, then turn.
    to_now = _rotation(-ego_yaw_rate * dt_s)
    ego_step = np.array([ego_speed * dt_s, 0.0])
    predicted = np.array(
        [to_now @ (t.xy + t.velocity * dt_s - ego_step) for t in tracks]
    ).reshape(-1, 2)
    updated = []
    for center in centers:
        match = None
        if len(tracks) > 0:
            distances = np.linalg.norm(predicted - center, axis=1)
            nearest = int(np.argmin(distances))
            gate = _NEW_TRACK_GATE_M if tracks[nearest].spins == 1 else _TRACK_GATE_M
            if distances[nearest] <= gate:
                match = tracks[nearest]
        if match is None:
            updated.append(_Track(center, np.zeros(2), np.zeros(2), 1))
            continue
        # Where it is now, back in the previous frame, minus where it was.
        moved = to_now.T @ center + ego_step - match.xy
        measured = to_now @ (moved / dt_s)
        if match.spins == 1:
            velocity, accel = measured, np.zeros(2)
        else:
            predicted_velocity = to_now @ (match.velocity + match.accel * dt_s)
            residual = measured - predicted_velocity
            velocity = predicted_velocity + _SPEED_GAIN * residual
            accel = np.clip(
                to_now @ match.accel + _ACCEL_GAIN * residual / dt_s,
                -_MAX_ACCEL_MPS2,
                _MAX_ACCEL_MPS2,
            )
        updated.append(_Track(center, velocity, accel, match.spins + 1))
    return updated


class LidarObstacleExtractor:
    """Stateful per-session extractor; the tracker associates across spins."""

    def __init__(
        self, min_confidence: float, min_range_m: float, max_range_m: float
    ) -> None:
        self._min_confidence = min_confidence
        self._min_range_m = min_range_m
        self._max_range_m = max_range_m
        self._detector = LidarDetector(config=DetectionConfig())
        self._tracks: list[_Track] = []
        self._last_timestamp_us: int | None = None

    def extract(
        self,
        spin: LidarSpin,
        ego_speed: float,
        ego_yaw_rate: float,
    ) -> list[dict[str, Any]]:
        """Detect obstacles in ``spin`` and return them in the rig frame, with
        their velocities relative to the road."""
        n = len(spin.xyz)
        if n == 0:
            return []

        # The pipeline's ground filter is a flat z cut around the sensor, so a
        # sensor tilted by even a degree would turn distant road into obstacles.
        # Rotate into the rig orientation but keep the sensor as the origin.
        sensor_in_rig = spin.rig_to_sensor.vec3.astype(np.float64)
        xyz_levelled = spin.sensor_to_rig(spin.xyz) - sensor_in_rig

        # SpinFrame columns: [x, y, z, intensity, ring, return_type]; the
        # payload has none of the last three (the detector classifies and
        # scores clusters by geometry only).
        points = np.zeros((n, 6), dtype=np.float32)
        points[:, :3] = xyz_levelled
        frame = SpinFrame(
            scan_index=0,
            timestamp_us=spin.timestamp_us,
            points=points,
            ego_pose=None,
        )
        detections = self._detector.detect(self._preprocessor(spin).process(frame))

        # Tracks follow each cluster's centroid; obstacles take its box.
        centroids = (
            np.array([obj.centroid for obj in detections.objects]).reshape(-1, 3)
            + sensor_in_rig
        )
        centers = np.array([obj.bbox.center for obj in detections.objects]).reshape(-1, 3)
        dt_s = (
            0.0
            if self._last_timestamp_us is None
            else (spin.timestamp_us - self._last_timestamp_us) * 1e-6
        )
        self._tracks = (
            _update_tracks(self._tracks, centroids[:, :2], ego_speed, ego_yaw_rate, dt_s)
            if dt_s > 0.0
            else [_Track(xy, np.zeros(2), np.zeros(2), 1) for xy in centroids[:, :2]]
        )
        self._last_timestamp_us = spin.timestamp_us

        centers_in_rig = centers + sensor_in_rig
        obstacles = []
        for obj, center_in_rig, track in zip(
            detections.objects, centers_in_rig, self._tracks
        ):
            if track.spins < _CONFIRM_SPINS:
                continue
            if obj.confidence < self._min_confidence:
                continue
            if not self._min_range_m <= obj.range_m <= self._max_range_m:
                continue

            obstacles.append(
                {
                    "id": obj.object_id,
                    "x": float(center_in_rig[0]),
                    "y": float(center_in_rig[1]),
                    "vx": float(track.velocity[0]),
                    "vy": float(track.velocity[1]),
                    "half_length_m": float(obj.bbox.extent[0]),
                    "half_width_m": float(obj.bbox.extent[1]),
                    "confidence": float(obj.confidence),
                    "source": "lidar",
                }
            )

        logger.debug("LiDAR: %d points -> %d obstacles", n, len(obstacles))
        return obstacles

    def _preprocessor(self, spin: LidarSpin) -> LidarPreprocessor:
        """Preprocess around the sensor, the origin the pipeline's ground and
        ego-shadow thresholds are written for.  The ground threshold assumes a
        sensor ``LidarExtrinsics().tz`` above the road; shift it so ground is cut
        at the same height above the road for this sensor's actual mount.
        """
        mount_height_m = float(spin.rig_to_sensor.vec3[2])
        defaults = PreprocessConfig()
        return LidarPreprocessor(
            config=PreprocessConfig(
                range_min=self._min_range_m,
                range_max=self._max_range_m,
                ground_z_threshold=defaults.ground_z_threshold
                + LidarExtrinsics().tz
                - mount_height_m,
                apply_motion_compensation=False,  # payload carries no ego pose
                apply_extrinsic_transform=False,  # detections are moved to rig below
                remove_ego_vehicle_shadow=True,
            )
        )
