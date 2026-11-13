# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_driver.radar_fusion import RadarConfirmer, fuse_radar
from alpasim_utils.radar_payload import RadarScan

_EGO_SPEED = 10.0


def _scan(*detections: tuple[float, float, float]) -> RadarScan:
    """Detections as (x, y, range_rate), seen from a radar at the rig origin."""
    xy = np.array([d[:2] for d in detections], dtype=np.float32).reshape(-1, 2)
    norms = np.linalg.norm(xy, axis=1, keepdims=True)
    return RadarScan(
        timestamp_us=0,
        xy=xy,
        range_rate=np.array([d[2] for d in detections], dtype=np.float32),
        line_of_sight=(xy / np.where(norms > 0, norms, 1.0)).astype(np.float32),
    )


def _lidar(x: float, y: float, vx: float) -> dict:
    return {
        "id": 1,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": 0.0,
        "confidence": 0.8,
        "source": "lidar",
    }


def _fuse(obstacles: list[dict], scan: RadarScan) -> list[dict]:
    return fuse_radar(
        obstacles,
        scan,
        np.ones(len(scan.xy), dtype=bool),
        _EGO_SPEED,
        association_gate_m=2.0,
        radar_only_confidence=0.5,
    )


def test_radar_range_rate_sets_speed_of_matched_lidar_obstacles() -> None:
    # The LiDAR detector split one stopped car into two clusters; the tracker's
    # speed guess is off.  Range rate -10 m/s at ego 10 m/s means stopped.
    obstacles = [_lidar(20.0, 0.0, vx=4.0), _lidar(21.5, 0.0, vx=3.0)]

    fused = _fuse(obstacles, _scan((20.2, 0.0, -10.0)))

    assert [o["vx"] for o in fused] == [pytest.approx(0.0, abs=1e-5)] * 2
    assert {o["source"] for o in fused} == {"lidar+radar"}


def test_unmatched_radar_detection_becomes_radar_only_obstacle() -> None:
    fused = _fuse([_lidar(20.0, 0.0, vx=0.0)], _scan((60.0, 0.0, -4.0)))

    (radar_only,) = [o for o in fused if o["source"] == "radar"]
    assert radar_only["x"] == pytest.approx(60.0)
    assert radar_only["vx"] == pytest.approx(6.0, abs=1e-5)
    assert radar_only["confidence"] == 0.5


def test_crossing_detection_speed_is_unobservable() -> None:
    """Straight to the side, range rate says nothing about x-speed: the LiDAR
    speed is kept, and a radar-only obstacle is treated as stationary."""
    lidar = _lidar(0.0, 5.0, vx=7.0)

    fused = _fuse([lidar], _scan((0.0, 5.0, 0.0), (0.0, -9.0, 0.0)))

    assert fused[0] == lidar
    assert fused[1]["source"] == "radar"
    assert fused[1]["vx"] == 0.0


def test_empty_scan_keeps_lidar_obstacles() -> None:
    obstacles = [_lidar(20.0, 0.0, vx=0.0)]

    assert _fuse(obstacles, _scan()) is obstacles


def _timed_scan(timestamp_us: int, xy: list[list[float]], range_rate: float = 0.0) -> RadarScan:
    return RadarScan(
        timestamp_us=timestamp_us,
        xy=np.array(xy, dtype=np.float32).reshape(-1, 2),
        range_rate=np.full(len(xy), range_rate, dtype=np.float32),
        line_of_sight=np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(xy), 1)),
    )


def test_confirmer_needs_a_detection_to_recur_near_its_predicted_spot() -> None:
    confirmer = RadarConfirmer(required_hits=2, window_scans=3, gate_m=2.0)

    assert not confirmer.update(_timed_scan(0, [[50.0, 0.0]], range_rate=-10.0)).any()
    # 10 m/s closing for 1 s: the same object is now at 40 m; the other is new.
    confirmed = confirmer.update(
        _timed_scan(1_000_000, [[40.0, 0.0], [20.0, 5.0]], range_rate=-10.0)
    )

    assert confirmed.tolist() == [True, False]


def test_confirmer_forgets_detections_older_than_its_window() -> None:
    confirmer = RadarConfirmer(required_hits=2, window_scans=2, gate_m=2.0)

    confirmer.update(_timed_scan(0, [[50.0, 0.0]]))
    confirmer.update(_timed_scan(100_000, []))

    assert not confirmer.update(_timed_scan(200_000, [[50.0, 0.0]])).any()


def test_two_radars_seeing_one_object_in_one_scan_do_not_confirm_it() -> None:
    confirmer = RadarConfirmer(required_hits=2, window_scans=3, gate_m=2.0)

    assert not confirmer.update(_timed_scan(0, [[50.0, 0.0], [50.1, 0.0]])).any()
