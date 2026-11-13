# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Fuse a radar scan into the LiDAR obstacles for the safety validator.

Radar measures range rate directly, while the LiDAR tracker can only
difference positions between spins.  Each LiDAR obstacle with a radar
detection within the association gate takes its speed from the nearest such
detection; one detection may serve several obstacles, since the LiDAR detector
can split one vehicle into several clusters.  Detections no LiDAR obstacle
explains become radar-only obstacles with a fixed, lower confidence, once
``RadarConfirmer`` has seen them recur: real objects persist between scans,
while clutter lands somewhere new each time.

Everything is in the rig frame, ego at the origin moving along +x.  Like the
LiDAR obstacles, a radar obstacle is assumed to move along x.  A radar-only
obstacle is the actor's nearest point, so it has no ``half_length_m`` or ``half_width_m``.
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
from alpasim_utils.radar_payload import RadarScan

# Below this |line of sight x|, the range rate says little about x-speed.
_MIN_LOS_X = 0.5


def _forward_speeds(scan: RadarScan, ego_speed: float) -> np.ndarray:
    """Obstacle x-speed per detection, NaN where unobservable.

    range_rate = (v_obstacle - v_ego) . line_of_sight with both velocities
    along x, so v_obstacle = v_ego + range_rate / los_x.
    """
    los_x = scan.line_of_sight[:, 0].astype(np.float64)
    observable = np.abs(los_x) >= _MIN_LOS_X
    speeds = np.full(len(los_x), np.nan)
    speeds[observable] = ego_speed + scan.range_rate[observable] / los_x[observable]
    return speeds


class RadarConfirmer:
    """Which detections of a scan recur in the recent scans; per-session state.

    A detection is confirmed when it and at least ``required_hits - 1`` of the
    previous ``window_scans - 1`` scans have a detection within ``gate_m`` of the
    same spot.  Earlier detections are moved along their line of sight by their
    range rate first.  Only earlier scans count, since several radars can see the
    same object in one scan.
    """

    def __init__(self, required_hits: int, window_scans: int, gate_m: float) -> None:
        self._required_hits = required_hits
        self._gate_m = gate_m
        self._previous: collections.deque[RadarScan] = collections.deque(
            maxlen=window_scans - 1
        )

    def update(self, scan: RadarScan) -> np.ndarray:
        """(N,) bool: the detections of ``scan`` that are confirmed."""
        hits = np.ones(len(scan.xy), dtype=int)
        for previous in self._previous:
            if len(scan.xy) == 0 or len(previous.xy) == 0:
                continue
            dt_s = (scan.timestamp_us - previous.timestamp_us) * 1e-6
            predicted = (
                previous.xy + previous.line_of_sight * previous.range_rate[:, None] * dt_s
            )
            distances = np.linalg.norm(scan.xy[:, None] - predicted[None], axis=2)
            hits += distances.min(axis=1) <= self._gate_m
        self._previous.append(scan)
        return hits >= self._required_hits


def fuse_radar(
    lidar_obstacles: list[dict[str, Any]],
    scan: RadarScan,
    confirmed: np.ndarray,
    ego_speed: float,
    association_gate_m: float,
    radar_only_confidence: float,
) -> list[dict[str, Any]]:
    """Return the LiDAR obstacles with radar speeds, plus the radar-only obstacles
    among the ``confirmed`` detections."""
    if len(scan.xy) == 0:
        return lidar_obstacles
    speeds = _forward_speeds(scan, ego_speed)
    explained = np.zeros(len(scan.xy), dtype=bool)

    fused = []
    for obstacle in lidar_obstacles:
        distances = np.linalg.norm(scan.xy - [obstacle["x"], obstacle["y"]], axis=1)
        explained |= distances <= association_gate_m
        nearest = int(np.argmin(distances))
        if distances[nearest] <= association_gate_m and not np.isnan(speeds[nearest]):
            obstacle = {
                **obstacle,
                "vx": float(speeds[nearest]),
                "source": "lidar+radar",
            }
        fused.append(obstacle)

    for i in np.flatnonzero(~explained & confirmed):
        fused.append(
            {
                "id": f"radar_{i}",
                "x": float(scan.xy[i, 0]),
                "y": float(scan.xy[i, 1]),
                # Treat an obstacle whose speed is unobservable as stationary.
                "vx": 0.0 if np.isnan(speeds[i]) else float(speeds[i]),
                "vy": 0.0,
                "confidence": radar_only_confidence,
                "source": "radar",
            }
        )
    return fused
