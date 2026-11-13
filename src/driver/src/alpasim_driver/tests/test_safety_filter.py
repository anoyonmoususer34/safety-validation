# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Any

import math

import numpy as np
import pytest
from alpasim_driver.models.base import ModelPrediction
from alpasim_driver.physics import compute_rss_max_safe_speed
from alpasim_driver.safety_filter import SafetyFilter, SensedObstacles
from alpasim_driver.scene_labels import SensorMode
from alpasim_driver.schema import SafetyValidatorConfig
from alpasim_utils.geometry import Pose
from alpasim_utils.lidar_payload import pack_lidar_spin
from alpasim_utils.radar_payload import RadarScan, pack_radar_scan
from alpasim_utils.sensor_payload import pack_sensor_payloads, unpack_sensor_payloads

_HZ = 10
_T = 40


# Roof LiDAR, as calibrated on the Hyperion rigs.
_LIDAR_IN_RIG = np.array([1.17, 0.0, 1.93])
_RIG_TO_LIDAR = Pose(
    _LIDAR_IN_RIG.astype(np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
)


def _payload(points_in_rig: np.ndarray, timestamp_us: int = 0) -> bytes:
    """Pack rig-frame points the way sensorsim returns them: in the sensor frame."""
    xyz = (points_in_rig - _LIDAR_IN_RIG).astype(np.float32)
    return pack_sensor_payloads(
        [
            pack_lidar_spin(
                xyz.tobytes(),
                len(xyz),
                timestamp_us,
                _RIG_TO_LIDAR,
            )
        ]
    )


def _car_ahead(distance_m: float, timestamp_us: int = 0) -> bytes:
    """Dense box of points, car-sized, centred on the ego path, in the rig frame."""
    rng = np.random.default_rng(0)
    n = 800
    return _payload(
        np.column_stack(
            [
                rng.uniform(distance_m, distance_m + 4.0, n),
                rng.uniform(-0.9, 0.9, n),
                rng.uniform(0.3, 1.8, n),
            ]
        ),
        timestamp_us,
    )


def _prediction(paths: list[np.ndarray]) -> ModelPrediction:
    positions = np.stack(paths)
    rotations = np.broadcast_to(np.eye(3), (*positions.shape[:2], 3, 3)).copy()
    return ModelPrediction(candidate_positions=positions, candidate_rotations=rotations)


def _straight(speed: float) -> np.ndarray:
    t = np.arange(1, _T + 1) / _HZ
    return np.column_stack([speed * t, np.zeros(_T), np.zeros(_T)])


def _swerve_left(speed: float) -> np.ndarray:
    """Turns hard left (~73 deg) after the first waypoint, out of the corridor."""
    path = _straight(speed)
    path[1:, 0] = path[0, 0] + 0.3 * np.arange(1, _T)
    path[1:, 1] = np.arange(1, _T, dtype=float)
    return path


def _filter(**cfg: Any) -> SafetyFilter:
    # Most tests judge single decisions; confirmation has tests of its own.
    cfg.setdefault("override_confirm_steps", 1)
    return SafetyFilter(
        SafetyValidatorConfig(enabled=True, **cfg), output_frequency_hz=_HZ
    )


def _step(
    safety_filter: SafetyFilter,
    prediction: ModelPrediction,
    renderer_data: bytes,
    speed: float,
    mode: SensorMode = SensorMode.FUSED,
) -> tuple[ModelPrediction, dict]:
    """One drive step: sense the forwarded sensors, then judge the candidates."""
    return safety_filter.apply(
        prediction, safety_filter.sense(renderer_data, speed, 0.0, mode), speed
    )


def test_clear_road_keeps_model_choice() -> None:
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])

    result, debug = _step(_filter(), prediction, _payload(np.zeros((0, 3))), 10.0)

    assert debug["reason"] == "safe"
    assert result.selected_index == 0
    np.testing.assert_array_equal(result.selected_positions, _straight(10.0))


def test_obstacle_ahead_slows_the_only_candidate() -> None:
    prediction = _prediction([_straight(10.0)])

    result, debug = _approach(_filter(), prediction, 8.0)

    assert debug["reason"] == "speed_override"
    assert debug["num_obstacles"] >= 1
    driven = np.linalg.norm(result.selected_positions[-1, :2])
    assert driven < np.linalg.norm(_straight(10.0)[-1, :2])


def test_obstacle_ahead_prefers_candidate_that_avoids_it() -> None:
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])
    safety_filter = _filter(max_switch_lateral_m=math.inf)
    # The tracker confirms the car after three spins: ego at 10 m/s closes 1 m
    # per 0.1 s.
    _step(safety_filter, prediction, _car_ahead(17.0, timestamp_us=0), 10.0)
    _step(safety_filter, prediction, _car_ahead(16.0, timestamp_us=100_000), 10.0)

    result, debug = _step(
        safety_filter, prediction, _car_ahead(15.0, timestamp_us=200_000), 10.0
    )

    assert debug["model_index"] == 0
    assert result.selected_index == 1


def test_missing_sensor_data_passes_prediction_through() -> None:
    prediction = _prediction([_straight(10.0)])

    result, debug = _step(_filter(), prediction, b"", 10.0)

    assert debug == {
        "reason": "no_sensor_data",
        "model_violations": [],
        "violations": [],
    }
    assert result is prediction


def test_radar_only_stopped_car_ahead_slows_down() -> None:
    """Without LiDAR, a radar detection of a stopped car in the path (closing
    at the ego speed) still makes the validator slow down."""
    radar = RadarScan(
        timestamp_us=0,
        xy=np.array([[8.0, 0.0]], dtype=np.float32),
        range_rate=np.array([-10.0], dtype=np.float32),
        line_of_sight=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    payload = pack_sensor_payloads([pack_radar_scan(radar)])

    safety_filter = _filter()
    prediction = _prediction([_straight(10.0)])
    _step(safety_filter, prediction, payload, 10.0)
    _, debug = _step(safety_filter, prediction, payload, 10.0)

    assert debug["reason"] == "speed_override"
    assert debug["num_radar_detections"] == 1
    assert "num_lidar_obstacles" not in debug


_NO_RADAR_DETECTIONS = RadarScan(
    timestamp_us=0,
    xy=np.zeros((0, 2), dtype=np.float32),
    range_rate=np.zeros(0, dtype=np.float32),
    line_of_sight=np.zeros((0, 2), dtype=np.float32),
)


def _with_empty_radar(lidar_payload: bytes) -> bytes:
    """Add a radar scan without detections to a LiDAR-only payload."""
    return pack_sensor_payloads(
        [
            *unpack_sensor_payloads(lidar_payload).values(),
            pack_radar_scan(_NO_RADAR_DETECTIONS),
        ]
    )


def test_sensors_off_keeps_model_plan_despite_obstacle() -> None:
    prediction = _prediction([_straight(10.0)])

    result, debug = _step(
        _filter(), prediction, _car_ahead(8.0), 10.0, mode=SensorMode.NONE
    )

    assert debug["reason"] == "sensors_off"
    assert result is prediction


def test_radar_mode_ignores_lidar_obstacle_that_fused_mode_brakes_for() -> None:
    prediction = _prediction([_straight(10.0)])

    def last_step(mode: SensorMode) -> dict:
        safety_filter = _filter()
        for timestamp_us, distance_m in ((0, 10.0), (100_000, 9.0), (200_000, 8.0)):
            payload = _with_empty_radar(_car_ahead(distance_m, timestamp_us))
            _, debug = _step(safety_filter, prediction, payload, 10.0, mode=mode)
        return debug

    fused = last_step(SensorMode.FUSED)
    radar = last_step(SensorMode.RADAR)

    assert fused["reason"] == "speed_override"
    assert radar["reason"] == "safe"
    assert "num_lidar_obstacles" not in radar


def test_policy_sensor_missing_from_payload_fails() -> None:
    radar_only = pack_sensor_payloads([pack_radar_scan(_NO_RADAR_DETECTIONS)])

    with pytest.raises(ValueError, match="LIDAR"):
        _filter().sense(radar_only, 10.0, 0.0, SensorMode.LIDAR)


def _approach(
    safety_filter: SafetyFilter, prediction: ModelPrediction, distance_m: float
) -> tuple[ModelPrediction, dict]:
    """Three spins 0.1 s apart closing on a stopped car at 10 m/s; the last step."""
    _step(safety_filter, prediction, _car_ahead(distance_m + 2.0, 0), 10.0)
    _step(safety_filter, prediction, _car_ahead(distance_m + 1.0, 100_000), 10.0)
    return _step(
        safety_filter, prediction, _car_ahead(distance_m, timestamp_us=200_000), 10.0
    )


def test_disabled_checks_let_the_model_plan_through() -> None:
    prediction = _prediction([_straight(10.0)])
    safety_filter = _filter(
        enable_emergency=False, enable_stopping_distance=False, enable_low_ttc=False
    )

    _, debug = _approach(safety_filter, prediction, 6.0)

    assert debug["reason"] == "safe"
    assert debug["violations"] == []


@pytest.mark.parametrize(
    ("distance_m", "reason", "violations"),
    [
        (16.0, "safe", []),  # bumper gap 11.9 m, TTC 1.2 s: only low-TTC would act
        (6.0, "speed_override", ["EMERGENCY_TTC_VIOLATION"]),  # gap 1.9 m, TTC 0.2 s
    ],
)
def test_emergency_only_brakes_just_below_emergency_ttc(
    distance_m: float, reason: str, violations: list[str]
) -> None:
    safety_filter = _filter(enable_stopping_distance=False, enable_low_ttc=False)

    _, debug = _approach(safety_filter, _prediction([_straight(10.0)]), distance_m)

    assert debug["reason"] == reason
    assert debug["violations"] == violations


def _radar_detection(x: float, y: float, range_rate: float) -> bytes:
    """One radar detection; range_rate is relative to the ego."""
    scan = RadarScan(
        timestamp_us=0,
        xy=np.array([[x, y]], dtype=np.float32),
        range_rate=np.array([range_rate], dtype=np.float32),
        line_of_sight=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    return pack_sensor_payloads([pack_radar_scan(scan)])


def _two_radar_steps(
    prediction: ModelPrediction, payload: bytes, speed: float, **cfg: Any
) -> tuple[ModelPrediction, dict]:
    """The second step, once the repeated detection is confirmed."""
    safety_filter = _filter(**cfg)
    _step(safety_filter, prediction, payload, speed, SensorMode.RADAR)
    return _step(safety_filter, prediction, payload, speed, SensorMode.RADAR)


def _shifted_left(speed: float, offset_m: float) -> np.ndarray:
    """A gentle lane change: straight speed, drifting left to ``offset_m``."""
    path = _straight(speed)
    path[:, 1] = offset_m * np.linspace(0.0, 1.0, _T) ** 2
    return path


def test_radar_detection_seen_once_is_clutter() -> None:
    """Different spots each scan never confirm; the same spot does on its second scan."""
    prediction = _prediction([_straight(10.0)])
    safety_filter = _filter()
    for x in (8.0, 30.0, 60.0):
        _, debug = _step(
            safety_filter,
            prediction,
            _radar_detection(x, 0.0, -10.0),
            10.0,
            SensorMode.RADAR,
        )
        assert debug["reason"] == "safe"


def test_candidate_that_changes_lane_is_chosen_over_one_that_hits_the_car_ahead() -> None:
    """The stopped car is 30 m ahead, where the second plan is 3.6 m to the left."""
    payload = _radar_detection(30.0, 0.0, -15.0)
    prediction = _prediction([_straight(15.0), _shifted_left(15.0, 15.0)])

    result, debug = _two_radar_steps(
        prediction, payload, 15.0, max_switch_lateral_m=math.inf
    )

    assert debug["model_violations"]
    assert result.selected_index == 1
    assert debug["reason"] == "safe"


def test_candidate_that_slows_down_is_chosen_over_one_at_the_current_speed() -> None:
    """A stopped car 24 m ahead of the bumper: too close for 15 m/s, not for 5 m/s."""
    payload = _radar_detection(28.0, 0.0, -15.0)
    prediction = _prediction([_straight(15.0), _straight(5.0)])

    result, debug = _two_radar_steps(prediction, payload, 15.0)

    assert debug["model_violations"]
    assert result.selected_index == 1
    assert debug["reason"] == "safe"


@pytest.mark.parametrize(
    ("range_rate", "reason"),
    [
        (0.0, "safe"),  # moving with the ego: it would brake too
        (-15.0, "speed_override"),  # stopped, at the same distance
    ],
)
def test_stopping_distance_counts_the_speed_of_the_vehicle_ahead(
    range_rate: float, reason: str
) -> None:
    """A 20 m bumper gap at 15 m/s: enough behind a moving car, not a stopped one."""
    payload = _radar_detection(24.0, 0.0, range_rate)

    _, debug = _two_radar_steps(_prediction([_straight(15.0)]), payload, 15.0)

    assert debug["reason"] == reason


def _obstacle(x: float, half_length_m: float = 0.0, vx: float = 0.0) -> dict[str, Any]:
    return {
        "id": 1,
        "x": x,
        "y": 0.0,
        "vx": vx,
        "vy": 0.0,
        "half_length_m": half_length_m,
        "confidence": 1.0,
    }


@pytest.mark.parametrize(
    ("half_length_m", "reason"),
    [
        (0.0, "safe"),
        (5.0, "speed_override"),  # a long truck's rear is 5 m nearer than its centre
    ],
)
def test_distance_is_measured_between_bumpers(
    half_length_m: float, reason: str
) -> None:
    """A plan at 15 m/s needs 28.25 m including the buffer; the car's centre is 30 m away."""
    _, debug = _filter().apply(
        _prediction([_straight(15.0)]),
        SensedObstacles(SensorMode.LIDAR, [_obstacle(30.0 + 4.093, half_length_m)], {}),
        15.0,
    )

    assert debug["reason"] == reason


@pytest.mark.parametrize(
    ("y", "reason"),
    [
        (2.4, "safe"),  # next lane: inside the corridor but clear of the ego's width
        (1.5, "speed_override"),  # overlaps the ego's width, a sideswipe
    ],
)
def test_slower_car_alongside_only_counts_if_it_overlaps_the_ego(
    y: float, reason: str
) -> None:
    alongside = {**_obstacle(3.0, half_length_m=2.0, vx=5.0), "y": y, "half_width_m": 0.9}

    _, debug = _filter().apply(
        _prediction([_straight(10.0)]),
        SensedObstacles(SensorMode.LIDAR, [alongside], {}),
        10.0,
    )

    assert debug["reason"] == reason


def _hold_stop_steps(
    detected_steps: int, clear_steps: int, ego_speed: float = 0.0
) -> list[str]:
    """The reasons for ``clear_steps`` empty scans after ``detected_steps`` with a
    stopped car 3 m ahead of a plan that would drive into it."""
    safety_filter = _filter()
    prediction = _prediction([_straight(10.0)])

    def step(obstacles: list[dict[str, Any]]) -> str:
        sensed = SensedObstacles(SensorMode.LIDAR, obstacles, {})
        return safety_filter.apply(prediction, sensed, ego_speed)[1]["reason"]

    for _ in range(detected_steps):
        step([_obstacle(4.093 + 3.0)])
    return [step([]) for _ in range(clear_steps)]


def test_one_false_alarm_does_not_hold_the_ego() -> None:
    assert _hold_stop_steps(detected_steps=1, clear_steps=3) == ["safe"] * 3


def test_hold_arms_only_once_the_ego_has_stopped() -> None:
    """At 10 m/s the ego is still braking: losing the car must not hold it."""
    assert _hold_stop_steps(detected_steps=3, clear_steps=3, ego_speed=10.0) == ["safe"] * 3


def test_hold_stops_after_its_limit_even_if_its_ghost_keeps_braking() -> None:
    """The ego keeps its plan's speed, so the ghost brakes it on every step."""
    reasons = _hold_stop_steps(detected_steps=2, clear_steps=60)

    assert reasons[0] == "speed_override"
    assert reasons[-1] == "safe"
    assert reasons.index("safe") == 49  # _HOLD_STOP_MAX_STEPS


def test_plan_braking_ahead_of_a_tailgater_is_not_a_violation() -> None:
    """The plan slows to 5 m/s from 10 m/s, with a car closing from 8 m behind."""
    tailgater = _obstacle(-8.0 - 1.3, vx=10.0)

    _, debug = _filter().apply(
        _prediction([_straight(5.0)]),
        SensedObstacles(SensorMode.LIDAR, [tailgater], {}),
        10.0,
    )

    assert debug["reason"] == "safe"
    assert debug["violations"] == []


def _curve_right(speed: float) -> np.ndarray:
    """A constant-radius 100 m bend to the right at ``speed``."""
    angle = speed * np.arange(1, _T + 1) / _HZ / 100.0
    return np.column_stack(
        [100.0 * np.sin(angle), -100.0 * (1.0 - np.cos(angle)), np.zeros(_T)]
    )


def test_braking_stays_on_the_plan_and_is_physically_followable() -> None:
    """Forced to a stop at 15 m/s: 6 m/s^2 needs 2.5 s and 18.75 m along the bend."""
    plan = _curve_right(15.0)
    obstacle = {
        "id": 1,
        "x": 4.093 + 0.5,  # right ahead of the bumper
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "confidence": 1.0,
    }

    result, debug = _filter().apply(
        _prediction([plan]), SensedObstacles(SensorMode.LIDAR, [obstacle], {}), 15.0
    )

    assert debug["safe_speed"] == 0.0
    driven = result.selected_positions
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros(3), driven]), axis=0), axis=1)
    assert steps[0] == pytest.approx(1.5 - 0.03, abs=0.01)  # 15 m/s, braking begun
    assert np.all(np.diff(steps) <= 1e-6)  # only ever slows
    assert steps.sum() == pytest.approx(18.75, abs=0.05)
    # Every waypoint is on the bend: at its distance from the bend's centre.
    centre = np.array([0.0, -100.0])
    radii = np.linalg.norm(driven[:, :2] - centre, axis=1)
    np.testing.assert_allclose(radii, 100.0, atol=0.01)


@pytest.mark.parametrize(
    ("vx", "reason"),
    [
        (0.0, "safe"),  # parked: braking cannot avoid it
        (5.0, "speed_override"),  # moving: it may be cutting in
    ],
)
def test_car_overlapping_alongside_only_counts_if_moving(vx: float, reason: str) -> None:
    alongside = {**_obstacle(3.0, half_length_m=2.0, vx=vx), "y": 1.5, "half_width_m": 0.9}

    _, debug = _filter().apply(
        _prediction([_straight(10.0)]),
        SensedObstacles(SensorMode.LIDAR, [alongside], {}),
        10.0,
    )

    assert debug["reason"] == reason


def test_car_merging_alongside_slows_the_ego_gently_not_with_an_emergency_stop() -> None:
    """Its tracked speed lags its real one, so it seems slower than the ego beside it."""
    merging = {**_obstacle(3.0, half_length_m=2.0, vx=9.0), "y": 1.0, "half_width_m": 0.9}

    result, debug = _filter().apply(
        _prediction([_straight(10.0)]),
        SensedObstacles(SensorMode.LIDAR, [merging], {}),
        10.0,
    )

    assert debug["reason"] == "speed_override"
    assert "EMERGENCY_TTC_VIOLATION" not in debug["violations"]
    first_step_m = float(np.linalg.norm(result.selected_positions[0, :2]))
    assert first_step_m > 1.0 - 0.5 * 6.0 * 0.01 + 1e-3


def test_obstacle_alongside_does_not_hold_the_ego_stopped() -> None:
    """A car cutting in alongside forces a stop; once it is gone, no ghost holds the ego."""
    safety_filter = _filter()
    cutting_in = {**_obstacle(3.0, half_length_m=2.0, vx=5.0), "y": 1.0, "half_width_m": 0.9}
    prediction = _prediction([_straight(10.0)])
    for _ in range(3):
        _, debug = safety_filter.apply(
            prediction, SensedObstacles(SensorMode.LIDAR, [cutting_in], {}), 10.0
        )
    assert debug["reason"] == "speed_override"

    _, debug = safety_filter.apply(
        prediction, SensedObstacles(SensorMode.LIDAR, [], {}), 10.0
    )

    assert debug["reason"] == "safe"


def test_car_parked_beside_the_lane_is_not_in_the_corridor() -> None:
    """Its near side is 2.5 m from the path: well clear of the ego's 1.05 m."""
    payload = _radar_detection(9.0, -2.5, -15.0)
    prediction = _prediction([_straight(15.0)])
    safety_filter = _filter()

    for _ in range(3):
        _, debug = _step(safety_filter, prediction, payload, 15.0, SensorMode.RADAR)

    assert debug["reason"] == "safe"


def test_one_flagged_step_does_not_slow_the_plan() -> None:
    """A stopped car 24 m ahead of the bumper is too close for 15 m/s but no emergency."""
    safety_filter = _filter(override_confirm_steps=2)
    prediction = _prediction([_straight(15.0)])
    sensed = SensedObstacles(SensorMode.LIDAR, [_obstacle(4.093 + 24.0)], {})

    reasons = [safety_filter.apply(prediction, sensed, 15.0)[1]["reason"] for _ in range(3)]

    assert reasons == ["unconfirmed", "speed_override", "speed_override"]


def test_an_emergency_brakes_without_waiting_for_confirmation() -> None:
    _, debug = _filter(override_confirm_steps=5).apply(
        _prediction([_straight(10.0)]),
        SensedObstacles(SensorMode.LIDAR, [_obstacle(4.093 + 3.0)], {}),
        10.0,
    )

    assert debug["reason"] == "speed_override"
    assert debug["violations"][0] == "EMERGENCY_TTC_VIOLATION"


@pytest.mark.parametrize(
    ("gap_m", "vx", "decel"),
    [
        (15.0, 12.0, 3.0),  # 15 -> 14.6 m/s: comfort braking
        (10.0, 10.0, 15.0 - 11.32),  # 15 -> 11.3 m/s within slowdown_time_s
        (24.0, 0.0, 6.0),  # a stop: 3 m/s^2 would need 37.5 m
        (5.0, 0.0, 6.0),  # an emergency stop
    ],
)
def test_a_slowdown_brakes_as_hard_as_reaching_the_safe_speed_takes(
    gap_m: float, vx: float, decel: float
) -> None:
    """At 15 m/s: the first 0.1 s covers 1.5 m less half a step of braking."""
    result, _ = _filter().apply(
        _prediction([_straight(15.0)]),
        SensedObstacles(SensorMode.LIDAR, [_obstacle(4.093 + gap_m, vx=vx)], {}),
        15.0,
    )

    first_step_m = float(np.linalg.norm(result.selected_positions[0, :2]))
    assert first_step_m == pytest.approx(1.5 - 0.5 * decel * 0.01, abs=1e-3)


def test_switching_logs_the_checks_the_model_candidate_failed() -> None:
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])

    _, debug = _approach(_filter(max_switch_lateral_m=math.inf), prediction, 15.0)

    assert debug["selected_index"] == 1
    assert debug["model_violations"]
    assert debug["violations"] == []


def test_rss_baseline_slows_the_model_candidate_without_switching() -> None:
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])

    _, debug = _approach(_filter(rss_baseline=True), prediction, 15.0)

    assert debug["reason"] == "speed_override"
    assert debug["selected_index"] == 0
    assert debug["violations"] == ["RSS_DISTANCE_VIOLATION"]
    assert 0.0 < debug["safe_speed"] < 10.0


def test_rss_baseline_keeps_plan_with_room_to_brake() -> None:
    prediction = _prediction([_straight(10.0)])

    _, debug = _approach(_filter(rss_baseline=True), prediction, 40.0)

    assert debug["reason"] == "safe"
    assert debug["violations"] == []


@pytest.mark.parametrize("front_speed", [0.0, 8.0])
def test_rss_safe_speed_meets_the_rss_distance_exactly(front_speed: float) -> None:
    rho, accel, min_brake, front_brake, gap = 0.5, 2.0, 4.0, 6.0, 20.0

    v = compute_rss_max_safe_speed(gap, front_speed, rho, accel, min_brake, front_brake)

    rss_distance = (
        v * rho
        + 0.5 * accel * rho**2
        + (v + rho * accel) ** 2 / (2 * min_brake)
        - front_speed**2 / (2 * front_brake)
    )
    assert rss_distance == pytest.approx(gap)


def _car_ahead_and_tailgater(ahead_m: float, behind_m: float, timestamp_us: int) -> bytes:
    rng = np.random.default_rng(0)

    def box(rear_x: float) -> np.ndarray:
        return np.column_stack(
            [
                rng.uniform(rear_x, rear_x + 4.0, 800),
                rng.uniform(-0.9, 0.9, 800),
                rng.uniform(0.3, 1.8, 800),
            ]
        )

    return _payload(np.vstack([box(ahead_m), box(behind_m)]), timestamp_us)


@pytest.mark.parametrize(
    ("rear_prediction_horizon", "violations", "keeps_speed"),
    [
        (30.0, ["LOW_TTC_VIOLATION", "REAR_COLLISION_RISK"], True),
        (0.0, ["LOW_TTC_VIOLATION"], False),  # abl_no_rear
    ],
)
def test_tailgater_keeps_speed_for_a_non_emergency_obstacle_ahead(
    rear_prediction_horizon: float, violations: list[str], keeps_speed: bool
) -> None:
    """A stopped car 17.9 m ahead (TTC 1.8 s) and a car 11.7 m behind closing at 10 m/s."""
    safety_filter = _filter(rear_prediction_horizon=rear_prediction_horizon)
    prediction = _prediction([_straight(10.0)])
    _step(safety_filter, prediction, _car_ahead_and_tailgater(24.0, -19.0, 0), 10.0)
    _step(safety_filter, prediction, _car_ahead_and_tailgater(23.0, -18.0, 100_000), 10.0)

    result, debug = _step(
        safety_filter, prediction, _car_ahead_and_tailgater(22.0, -17.0, 200_000), 10.0
    )

    assert debug["violations"] == violations
    driven = np.linalg.norm(result.selected_positions[-1, :2])
    planned = np.linalg.norm(_straight(10.0)[-1, :2])
    assert bool(driven > 0.99 * planned) == keeps_speed


def _pedestrian(gap_m: float, y: float, vy: float) -> dict[str, Any]:
    return {
        **_obstacle(4.093 + gap_m, half_length_m=0.3),
        "y": y,
        "vy": vy,
        "half_width_m": 0.3,
    }


def test_pedestrian_walking_toward_the_path_slows_the_ego() -> None:
    """At 10 m/s the ego reaches them in 2 s, when they have walked 3 m onto its path."""
    _, debug = _filter().apply(
        _prediction([_straight(10.0)]),
        SensedObstacles(SensorMode.LIDAR, [_pedestrian(20.0, -3.0, 1.5)], {}),
        10.0,
    )

    assert debug["reason"] == "speed_override"
    (relevant,) = debug["validator_result"]["trace_data"]["relevant_obstacles"]
    assert relevant["crossing"]


@pytest.mark.parametrize(
    ("pedestrian", "cfg"),
    [
        (_pedestrian(20.0, -3.0, -1.5), {}),  # walking away from the path
        (_pedestrian(20.0, -3.0, 0.3), {}),  # sideways jitter, not walking
        (_pedestrian(30.0, -3.0, 4.0), {}),  # across and clear before the ego arrives
        (_pedestrian(20.0, -3.0, 1.5), {"crossing_horizon_s": 0.0}),
    ],
)
def test_pedestrian_who_will_not_be_in_the_path_is_ignored(
    pedestrian: dict[str, Any], cfg: dict[str, Any]
) -> None:
    _, debug = _filter(**cfg).apply(
        _prediction([_straight(10.0)]),
        SensedObstacles(SensorMode.LIDAR, [pedestrian], {}),
        10.0,
    )

    assert debug["reason"] == "safe"


def test_obstacle_ahead_brakes_in_lane_rather_than_swerving_onto_another_plan() -> None:
    """The only plan that avoids the car leaves the lane: the model's own is braked."""
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])

    result, debug = _approach(_filter(), prediction, 15.0)

    assert result.selected_index == 0
    assert debug["reason"] == "speed_override"



@pytest.mark.parametrize(("fused_radar_only_emergency", "emergency"), [(False, False), (True, True)])
def test_radar_only_detection_cannot_stop_hard_when_lidar_runs(
    fused_radar_only_emergency: bool, emergency: bool
) -> None:
    """A radar detection 2 m ahead that no LiDAR return explains, closing at the
    ego speed: with LiDAR running it slows the ego but is no emergency."""
    radar = RadarScan(
        timestamp_us=0,
        xy=np.array([[4.093 + 2.0, 0.0]], dtype=np.float32),
        range_rate=np.array([-10.0], dtype=np.float32),
        line_of_sight=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    payload = pack_sensor_payloads(
        [*unpack_sensor_payloads(_payload(np.zeros((0, 3)))).values(), pack_radar_scan(radar)]
    )
    safety_filter = _filter(fused_radar_only_emergency=fused_radar_only_emergency)
    prediction = _prediction([_straight(10.0)])
    _step(safety_filter, prediction, payload, 10.0)
    _, debug = _step(safety_filter, prediction, payload, 10.0)

    assert debug["reason"] == "speed_override"
    assert ("EMERGENCY_TTC_VIOLATION" in debug["violations"]) is emergency
