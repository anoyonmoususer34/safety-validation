# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""LiDAR/radar safety filter over the model's sampled trajectory candidates.

Obstacles come from whichever sensors the runtime forwards: LiDAR detections,
radar detections, or LiDAR detections with radar speeds (``radar_fusion``).

Candidates are checked in model-preference order (the model's own selection
first), each on its own speed and path; only those within
``max_switch_lateral_m`` sideways of the model's own plan are considered.  The first one the ``SafetyValidator``
accepts is driven as-is.  If none pass, the candidate with the highest validated
safe speed is driven, braking along its own path to that speed at
``override_decel``.  Except in an
emergency, it is slowed only once the model's own plan has failed
``override_confirm_steps`` Drive calls in a row; until then the plan is driven
unchanged (reason ``unconfirmed``).

With ``rss_baseline`` only the model's own candidate is checked, against the
RSS longitudinal safe distance, and slowed to the RSS safe speed if needed.

The debug summary lists the checks the model's own candidate failed
(``model_violations``) and those of the driven candidate (``violations``).

Everything runs in the rig frame of the model's t0: ego at the origin with
heading 0, obstacles and candidate waypoints in that same frame.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from typing import Any

import numpy as np
from alpasim_utils import lidar_payload, radar_payload
from alpasim_utils.sensor_payload import unpack_sensor_payloads

from .lidar_obstacle_extractor import LidarObstacleExtractor
from .models.base import ModelPrediction
from .radar_fusion import RadarConfirmer, fuse_radar
from .safety_validator import SafetyValidator
from .scene_labels import SensorMode
from .schema import SafetyValidatorConfig

logger = logging.getLogger(__name__)


_PLAN_SPEED_WINDOW_S = 1.0  # a plan's speed is its average over this long


def _plan_speed(xy: np.ndarray, output_frequency_hz: int) -> float:
    """Average speed the (T, 2) rig-frame waypoints drive from the rig origin."""
    n = max(int(_PLAN_SPEED_WINDOW_S * output_frequency_hz), 1)
    path = np.vstack([[0.0, 0.0], xy[:n]])
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.mean(segments)) * output_frequency_hz


def _lateral_departure(xy: np.ndarray, reference_xy: np.ndarray) -> float:
    """Largest sideways distance of the (T, 2) path ``xy`` from ``reference_xy``,
    each point compared with the reference at the same distance ahead."""
    forward = np.maximum.accumulate(reference_xy[:, 0])
    return float(np.max(np.abs(xy[:, 1] - np.interp(xy[:, 0], forward, reference_xy[:, 1]))))


def _action_from_rig_xy(xy: np.ndarray, output_frequency_hz: int) -> dict[str, Any]:
    """The plan's speed and path.  Steering, for traces, is the heading change
    between the first two waypoint segments."""
    heading0 = math.atan2(float(xy[0, 1]), float(xy[0, 0]))
    steering_angle = 0.0
    if len(xy) >= 2:
        dx, dy = xy[1] - xy[0]
        steering_angle = math.atan2(float(dy), float(dx)) - heading0
    return {
        "target_speed": _plan_speed(xy, output_frequency_hz),
        "steering_angle": steering_angle,
        "path_xy": xy,
    }


def override_decel(
    cfg: SafetyValidatorConfig, speed: float, safe_speed: float, emergency: bool
) -> float:
    """m/s^2 a speed override brakes at: ``max_decel`` in an emergency, otherwise
    the rate that reaches ``safe_speed`` within ``slowdown_time_s``, clipped to
    [``comfort_decel``, ``max_decel``]."""
    if emergency:
        return cfg.max_decel
    needed = (speed - safe_speed) / cfg.slowdown_time_s
    return min(max(needed, cfg.comfort_decel), cfg.max_decel)


def _brake_along_path(
    positions: np.ndarray,
    rotations: np.ndarray,
    speed: float,
    target_speed: float,
    output_frequency_hz: int,
    max_decel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(T, 3) waypoints and (T, 3, 3) rotations that slow to ``target_speed``.

    They stay on the plan's own path, at a deceleration the vehicle can follow
    from ``speed``: the controller steers hard toward waypoints it cannot reach.
    A plan already no faster than ``target_speed`` is returned unchanged.
    """
    if _plan_speed(positions[:, :2], output_frequency_hz) <= target_speed:
        return positions, rotations
    times = np.arange(1, len(positions) + 1) / output_frequency_hz
    start_speed = max(speed, target_speed)
    brake_time = (start_speed - target_speed) / max_decel
    braking = np.minimum(times, brake_time)
    travelled = (
        start_speed * braking
        - 0.5 * max_decel * braking**2
        + target_speed * (times - braking)
    )
    path = np.vstack([np.zeros(3), positions])
    path_length = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    )
    # Never ahead of where the plan itself would be by then.
    travelled = np.minimum(travelled, path_length[1:])
    braked = np.column_stack(
        [np.interp(travelled, path_length, path[:, axis]) for axis in range(3)]
    )
    nearest = np.searchsorted(path_length, travelled).clip(0, len(positions))
    path_rotations = np.concatenate([np.eye(3)[None], rotations])
    return braked, path_rotations[nearest]


@dataclasses.dataclass(frozen=True)
class SensedObstacles:
    """Rig-frame obstacles for one step, plus per-sensor debug counts."""

    mode: SensorMode
    obstacles: list[dict[str, Any]]
    debug: dict[str, Any]


class SafetyFilter:
    """Per-session state: the LiDAR tracker, the radar confirmer and the
    validator's hold-stop latch."""

    def __init__(self, cfg: SafetyValidatorConfig, output_frequency_hz: int) -> None:
        self._cfg = cfg
        self._output_frequency_hz = output_frequency_hz
        self._extractor = LidarObstacleExtractor(
            min_confidence=cfg.min_confidence,
            min_range_m=cfg.lidar_min_range_m,
            max_range_m=cfg.lidar_max_range_m,
        )
        self._radar_confirmer = RadarConfirmer(
            required_hits=cfg.radar_confirm_hits,
            window_scans=cfg.radar_confirm_scans,
            gate_m=cfg.radar_confirm_gate_m,
        )
        # Drive calls in a row on which the model's own plan failed a check.
        self._model_flagged_steps = 0
        self._validator = SafetyValidator(
            lateral_margin_m=cfg.lateral_margin_m,
            prediction_horizon=cfg.prediction_horizon,
            reaction_time=cfg.reaction_time,
            max_decel=cfg.max_decel,
            min_ttc=cfg.min_ttc,
            emergency_ttc=cfg.emergency_ttc,
            safety_buffer=cfg.safety_buffer,
            min_confidence=cfg.min_confidence,
            confidence_weight_power=cfg.confidence_weight_power,
            rear_min_ttc=cfg.rear_min_ttc,
            rear_emergency_ttc=cfg.rear_emergency_ttc,
            rear_prediction_horizon=cfg.rear_prediction_horizon,
            enable_emergency=cfg.enable_emergency,
            enable_stopping_distance=cfg.enable_stopping_distance,
            enable_low_ttc=cfg.enable_low_ttc,
            ego_front_m=cfg.ego_front_m,
            ego_rear_m=cfg.ego_rear_m,
            ego_half_width_m=cfg.ego_half_width_m,
            crossing_horizon_s=cfg.crossing_horizon_s,
            crossing_min_lateral_speed=cfg.crossing_min_lateral_speed_mps,
        )

    def sense(
        self, renderer_data: bytes, speed: float, yaw_rate: float, mode: SensorMode
    ) -> SensedObstacles | None:
        """Obstacles from the sensors ``mode`` selects; None when none arrived.
        ``speed`` (m/s) and ``yaw_rate`` (rad/s) are the ego's, to take its own
        motion out of the tracked obstacles' velocities.

        Independent of the model's prediction, so the driver runs it while the
        model infers.  Calls must stay in step order: the LiDAR tracker and the
        radar confirmer compare each sweep with the previous ones.
        """
        if mode is SensorMode.NONE:
            return SensedObstacles(mode=mode, obstacles=[], debug={})
        if not renderer_data:
            # The runtime forwards sensor data once per camera render; drive
            # calls that outnumber renders (or force-GT cache replays) carry none.
            logger.warning("Safety validator enabled but no sensor data received")
            return None
        obstacles, debug = self._obstacles(renderer_data, speed, yaw_rate, mode)
        return SensedObstacles(mode=mode, obstacles=obstacles, debug=debug)

    def apply(
        self,
        prediction: ModelPrediction,
        sensed: SensedObstacles | None,
        speed: float,
    ) -> tuple[ModelPrediction, dict[str, Any]]:
        """Return the prediction to drive and a debug summary of the decision."""
        if sensed is None:
            return prediction, {
                "reason": "no_sensor_data",
                "model_violations": [],
                "violations": [],
            }
        if sensed.mode is SensorMode.NONE:
            return prediction, {
                "reason": "sensors_off",
                "sensor_mode": "none",
                "model_violations": [],
                "violations": [],
            }

        obstacles, sensor_debug = sensed.obstacles, sensed.debug
        ego_state = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": speed}
        if self._cfg.rss_baseline:
            return self._apply_rss(prediction, obstacles, sensor_debug, ego_state)

        k = len(prediction.candidate_positions)
        model_xy = prediction.candidate_positions[prediction.selected_index, :, :2]
        order = [prediction.selected_index] + [
            i for i in range(k)
            if i != prediction.selected_index
            and _lateral_departure(prediction.candidate_positions[i, :, :2], model_xy)
            <= self._cfg.max_switch_lateral_m
        ]
        # The validator carries a hold-stop latch across steps.  Every candidate
        # is judged from the same pre-step latch state, and only the driven
        # candidate's resulting state is kept.
        results = {}
        validators = {}
        for i in order:
            action = _action_from_rig_xy(
                prediction.candidate_positions[i, :, :2], self._output_frequency_hz
            )
            validators[i] = copy.copy(self._validator)
            result = validators[i].validate(ego_state, action, obstacles)
            results[i] = result
            if i == prediction.selected_index:
                self._model_flagged_steps = (
                    0 if result["is_safe"] else self._model_flagged_steps + 1
                )
            if result["is_safe"]:
                self._validator = validators[i]
                return dataclasses.replace(prediction, selected_index=i), {
                    "reason": "safe",
                    "model_index": prediction.selected_index,
                    "selected_index": i,
                    "model_violations": results[prediction.selected_index][
                        "violations"
                    ],
                    "violations": result["violations"],
                    "num_obstacles": len(obstacles),
                    **sensor_debug,
                    "validator_result": result,
                }

        best_i = max(
            order, key=lambda i: results[i]["validated_action"]["target_speed"]
        )
        emergency = "EMERGENCY_TTC_VIOLATION" in results[best_i]["violations"]
        if not emergency and self._model_flagged_steps < self._cfg.override_confirm_steps:
            # Not yet confirmed: drive the model's plan, keeping the latch as it was.
            return prediction, {
                "reason": "unconfirmed",
                "model_index": prediction.selected_index,
                "selected_index": prediction.selected_index,
                "model_violations": results[prediction.selected_index]["violations"],
                "violations": [],
                "num_obstacles": len(obstacles),
                **sensor_debug,
                "validator_result": results[prediction.selected_index],
            }
        self._validator = validators[best_i]
        safe_speed = results[best_i]["validated_action"]["target_speed"]
        positions = prediction.candidate_positions.copy()
        rotations = prediction.candidate_rotations.copy()
        positions[best_i], rotations[best_i] = _brake_along_path(
            positions[best_i],
            rotations[best_i],
            speed,
            safe_speed,
            self._output_frequency_hz,
            override_decel(self._cfg, speed, safe_speed, emergency),
        )
        logger.info(
            "No safe candidate among %d; driving %d at %.2f m/s (violations=%s)",
            k,
            best_i,
            safe_speed,
            results[best_i]["violations"],
        )
        return dataclasses.replace(
            prediction,
            candidate_positions=positions,
            candidate_rotations=rotations,
            selected_index=best_i,
        ), {
            "reason": "speed_override",
            "model_index": prediction.selected_index,
            "selected_index": best_i,
            "safe_speed": safe_speed,
            "model_violations": results[prediction.selected_index]["violations"],
            "violations": results[best_i]["violations"],
            "num_obstacles": len(obstacles),
            **sensor_debug,
            "validator_result": results[best_i],
        }

    def _apply_rss(
        self,
        prediction: ModelPrediction,
        obstacles: list[dict[str, Any]],
        sensor_debug: dict[str, Any],
        ego_state: dict[str, float],
    ) -> tuple[ModelPrediction, dict[str, Any]]:
        """Slow the model's own candidate to the RSS safe speed if it exceeds it."""
        i = prediction.selected_index
        speed = ego_state["speed"]
        action = _action_from_rig_xy(
            prediction.candidate_positions[i, :, :2], self._output_frequency_hz
        )
        safe_speed, assessments = self._validator.rss_safe_speed(
            ego_state,
            action,
            obstacles,
            max_accel=self._cfg.rss_max_accel,
            min_brake=self._cfg.rss_min_brake,
        )
        debug = {
            "model_index": i,
            "selected_index": i,
            "num_obstacles": len(obstacles),
            **sensor_debug,
            "rss_assessments": assessments,
        }
        if safe_speed >= speed:
            return prediction, {
                "reason": "safe",
                "model_violations": [],
                "violations": [],
                **debug,
            }
        positions = prediction.candidate_positions.copy()
        rotations = prediction.candidate_rotations.copy()
        positions[i], rotations[i] = _brake_along_path(
            positions[i],
            rotations[i],
            speed,
            safe_speed,
            self._output_frequency_hz,
            self._cfg.rss_min_brake,
        )
        return dataclasses.replace(
            prediction, candidate_positions=positions, candidate_rotations=rotations
        ), {
            "reason": "speed_override",
            "safe_speed": safe_speed,
            "model_violations": ["RSS_DISTANCE_VIOLATION"],
            "violations": ["RSS_DISTANCE_VIOLATION"],
            **debug,
        }

    def _obstacles(
        self, renderer_data: bytes, speed: float, yaw_rate: float, mode: SensorMode
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rig-frame obstacles from the selected sensors, plus debug counts.

        FUSED uses whichever sensors were forwarded; LIDAR and RADAR require
        that sensor, since the runtime must forward what the policy selects.
        """
        payloads = unpack_sensor_payloads(renderer_data)
        if mode is SensorMode.FUSED:
            use_lidar = lidar_payload.MAGIC in payloads
            use_radar = radar_payload.MAGIC in payloads
        else:
            use_lidar = mode is SensorMode.LIDAR
            use_radar = mode is SensorMode.RADAR
            required = lidar_payload.MAGIC if use_lidar else radar_payload.MAGIC
            if required not in payloads:
                raise ValueError(
                    f"Safety sensor mode {mode} needs {required.decode()} data, "
                    "which the runtime did not forward"
                )
        obstacles: list[dict[str, Any]] = []
        debug: dict[str, Any] = {"sensor_mode": str(mode)}
        if use_lidar:
            spin = lidar_payload.unpack_lidar_spin(payloads[lidar_payload.MAGIC])
            obstacles = self._extractor.extract(spin, speed, yaw_rate)
            debug["num_lidar_obstacles"] = len(obstacles)
        if use_radar:
            scan = radar_payload.unpack_radar_scan(payloads[radar_payload.MAGIC])
            obstacles = fuse_radar(
                obstacles,
                scan,
                self._radar_confirmer.update(scan),
                speed,
                association_gate_m=self._cfg.radar_association_gate_m,
                radar_only_confidence=self._cfg.radar_only_confidence,
            )
            debug["num_radar_detections"] = len(scan.xy)
        # Radar alone produces ghost detections; with LiDAR running, an obstacle no
        # LiDAR detection explains slows the ego but cannot make it stop hard.
        if use_lidar and use_radar and not self._cfg.fused_radar_only_emergency:
            obstacles = [
                {**obs, "may_emergency_brake": obs["source"] != "radar"} for obs in obstacles
            ]
        return obstacles, debug
