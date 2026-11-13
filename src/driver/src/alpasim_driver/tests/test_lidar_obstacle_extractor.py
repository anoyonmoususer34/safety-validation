# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Any

import numpy as np
import pytest
from alpasim_driver.lidar_obstacle_extractor import LidarObstacleExtractor
from alpasim_utils.geometry import Pose
from alpasim_utils.lidar_payload import pack_lidar_spin, unpack_lidar_spin

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
# 1 deg nose-down pitch, about the tilt of the recorded Hyperion roof LiDARs.
_PITCHED = np.array(
    [0.0, np.sin(np.radians(0.5)), 0.0, np.cos(np.radians(0.5))], dtype=np.float32
)


def _mount(lidar_in_rig: list[float], rotation_xyzw: np.ndarray = _IDENTITY) -> Pose:
    return Pose(np.array(lidar_in_rig, dtype=np.float32), rotation_xyzw)


def _extract(points_in_rig: np.ndarray, rig_to_lidar: Pose) -> list[dict]:
    """Run the extractor on rig-frame points seen by a LiDAR at ``rig_to_lidar``."""
    lidar_from_rig = np.linalg.inv(rig_to_lidar.as_se3())
    xyz = (points_in_rig @ lidar_from_rig[:3, :3].T + lidar_from_rig[:3, 3]).astype(
        np.float32
    )
    extractor = LidarObstacleExtractor(
        min_confidence=0.1, min_range_m=2.0, max_range_m=80.0
    )
    # A stopped ego and static points: seen in enough spins to count as obstacles.
    for timestamp_us in (0, 100_000, 200_000):
        payload = pack_lidar_spin(xyz.tobytes(), len(xyz), timestamp_us, rig_to_lidar)
        obstacles = extractor.extract(
            unpack_lidar_spin(payload), ego_speed=0.0, ego_yaw_rate=0.0
        )
    return obstacles


def _road(rig_to_lidar: Pose) -> np.ndarray:
    """Rig-frame road returns of a spinning LiDAR: one ring per downward beam,
    ray-cast onto the z = 0 road plane, with a few cm of surface roughness."""
    elevations = np.radians(np.linspace(-25.0, -1.0, 64))
    azimuths = np.linspace(-np.pi, np.pi, 1800, endpoint=False)
    elevation, azimuth = np.meshgrid(elevations, azimuths, indexing="ij")
    rays_in_lidar = np.stack(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=-1,
    ).reshape(-1, 3)
    rig_from_lidar = rig_to_lidar.as_se3().astype(np.float64)
    rays_in_rig = rays_in_lidar @ rig_from_lidar[:3, :3].T
    origin_in_rig = rig_from_lidar[:3, 3]
    distance = -origin_in_rig[2] / rays_in_rig[:, 2]
    hits = (distance > 0) & (distance < 80.0)
    road = origin_in_rig + rays_in_rig[hits] * distance[hits, None]
    road[:, 2] += np.random.default_rng(0).normal(0.0, 0.02, len(road))
    return road


def _car(rng: np.random.Generator, rear_x: float) -> np.ndarray:
    n = 800
    return np.column_stack(
        [
            np.full(n, rear_x),  # rear face, as seen from behind
            rng.uniform(-0.9, 0.9, n),
            rng.uniform(0.3, 1.5, n),
        ]
    )


def test_road_returns_are_not_obstacles() -> None:
    for mount in (
        _mount([1.17, 0.0, 1.93]),
        _mount([0.5, 0.0, 2.37]),
        _mount([1.17, 0.0, 1.93], _PITCHED),
    ):
        assert _extract(_road(mount), mount) == []


def test_obstacles_are_reported_in_rig_frame() -> None:
    mount = _mount([1.17, 0.0, 1.93], _PITCHED)
    points_in_rig = np.vstack([_road(mount), _car(np.random.default_rng(0), 15.0)])

    (obstacle,) = _extract(points_in_rig, mount)

    assert abs(obstacle["x"] - 15.0) < 0.3
    assert abs(obstacle["y"]) < 0.3


def test_approaching_objects_ahead_are_slower_and_behind_are_faster() -> None:
    """Ego at 10 m/s; one car ahead and one behind each close in at 5 m/s."""
    mount = _mount([1.17, 0.0, 1.93])
    extractor = LidarObstacleExtractor(
        min_confidence=0.1, min_range_m=2.0, max_range_m=80.0
    )
    rng = np.random.default_rng(0)
    lidar_from_rig = np.linalg.inv(mount.as_se3())

    def spin(ahead_x: float, behind_x: float, timestamp_us: int) -> list[dict]:
        behind = _car(rng, behind_x)  # its front face, as seen from ahead
        points_in_rig = np.vstack([_car(rng, ahead_x), behind])
        xyz = points_in_rig @ lidar_from_rig[:3, :3].T + lidar_from_rig[:3, 3]
        payload = pack_lidar_spin(
            xyz.astype(np.float32).tobytes(), len(xyz), timestamp_us, mount
        )
        return extractor.extract(unpack_lidar_spin(payload), ego_speed=10.0, ego_yaw_rate=0.0)

    assert spin(15.5, -14.5, 0) == []
    assert spin(15.0, -14.0, 100_000) == []
    obstacles = spin(14.5, -13.5, 200_000)

    by_side = {"ahead" if o["x"] > 0 else "behind": o["vx"] for o in obstacles}
    assert by_side["ahead"] == pytest.approx(5.0, abs=1.0)
    assert by_side["behind"] == pytest.approx(15.0, abs=1.0)


def test_cluster_seen_once_is_not_an_obstacle() -> None:
    """Clusters in different places each spin never form a track."""
    mount = _mount([1.17, 0.0, 1.93])
    extractor = LidarObstacleExtractor(
        min_confidence=0.1, min_range_m=2.0, max_range_m=80.0
    )
    rng = np.random.default_rng(0)
    lidar_from_rig = np.linalg.inv(mount.as_se3())
    for i, rear_x in enumerate((10.0, 25.0, 40.0, 55.0)):
        points = _car(rng, rear_x)
        xyz = points @ lidar_from_rig[:3, :3].T + lidar_from_rig[:3, 3]
        payload = pack_lidar_spin(
            xyz.astype(np.float32).tobytes(), len(xyz), i * 100_000, mount
        )
        assert extractor.extract(unpack_lidar_spin(payload), ego_speed=0.0, ego_yaw_rate=0.0) == []


def _block(rng: np.random.Generator, rear_x: float, center_y: float) -> np.ndarray:
    """A car's rear face, as seen from behind, centred ``center_y`` to the left."""
    n = 800
    return np.column_stack(
        [
            np.full(n, rear_x),
            rng.uniform(center_y - 0.9, center_y + 0.9, n),
            rng.uniform(0.3, 1.5, n),
        ]
    )


def _track_world_block(
    world_xy_at: Any, ego_speed: float, ego_yaw_rate: float, spins: int = 6
) -> list[dict]:
    """Obstacles after ``spins`` spins of a block whose rear-face centre is at
    ``world_xy_at(i)`` in spin i, seen by an ego driving and turning from the origin."""
    mount = _mount([1.17, 0.0, 1.93])
    lidar_from_rig = np.linalg.inv(mount.as_se3())
    extractor = LidarObstacleExtractor(
        min_confidence=0.1, min_range_m=2.0, max_range_m=80.0
    )
    rng = np.random.default_rng(0)
    dt_s = 0.1
    ego_xy, ego_yaw = np.zeros(2), 0.0
    for i in range(spins):
        world = _block(rng, 0.0, 0.0)
        world[:, :2] += world_xy_at(i)
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        in_rig = world.copy()
        in_rig[:, :2] = (world[:, :2] - ego_xy) @ np.array([[c, -s], [s, c]])
        xyz = in_rig @ lidar_from_rig[:3, :3].T + lidar_from_rig[:3, 3]
        payload = pack_lidar_spin(
            xyz.astype(np.float32).tobytes(), len(xyz), i * 100_000, mount
        )
        obstacles = extractor.extract(
            unpack_lidar_spin(payload), ego_speed=ego_speed, ego_yaw_rate=ego_yaw_rate
        )
        # Drive forward, then turn: the motion the tracker takes out.
        ego_xy = ego_xy + ego_speed * dt_s * np.array([c, s])
        ego_yaw += ego_yaw_rate * dt_s
    return obstacles


def test_a_turning_ego_sees_a_parked_car_as_stationary() -> None:
    """Turning at 0.2 rad/s, a car 20 m ahead sweeps across at about 4 m/s in the rig frame."""
    (obstacle,) = _track_world_block(
        lambda i: np.array([20.0, 2.0]), ego_speed=10.0, ego_yaw_rate=0.2
    )
    assert obstacle["vx"] == pytest.approx(0.0, abs=0.5)
    assert obstacle["vy"] == pytest.approx(0.0, abs=0.5)


def test_a_crossing_object_has_its_sideways_speed() -> None:
    (obstacle,) = _track_world_block(
        lambda i: np.array([15.0, -4.0 + 1.5 * 0.1 * i]), ego_speed=0.0, ego_yaw_rate=0.0
    )
    assert obstacle["vx"] == pytest.approx(0.0, abs=0.3)
    assert obstacle["vy"] == pytest.approx(1.5, abs=0.3)


def test_spin_with_no_return_in_range_has_no_obstacles() -> None:
    """Every return closer than min_range_m, e.g. a pedestrian beside the sensor."""
    mount = _mount([1.17, 0.0, 1.93])
    xyz = np.random.default_rng(0).uniform(-0.5, 0.5, (50, 3)).astype(np.float32)
    payload = pack_lidar_spin(xyz.tobytes(), len(xyz), 0, mount)
    extractor = LidarObstacleExtractor(
        min_confidence=0.1, min_range_m=2.0, max_range_m=80.0
    )
    assert extractor.extract(unpack_lidar_spin(payload), ego_speed=5.0, ego_yaw_rate=0.0) == []
