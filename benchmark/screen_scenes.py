"""Rank a scene suite by how close its logged traffic comes to the ego vehicle.

Closed-loop collisions happen where the policy deviates from the log in tight
traffic: a short headway to the vehicle ahead, a vehicle close behind that
rear-ends the ego if it brakes, or a vehicle passing close alongside.  This
measures those margins on the logged ego trajectory, so it needs only each
scene's tracks, not its neural reconstruction: it reads metadata.yaml,
rig_trajectories.json and sequence_tracks.json out of the remote USDZ (about
12 MB of 1.9 GB) and builds a local stub artifact from them.

Results are appended to OUTPUT as they arrive; rerunning skips scenes already
in it.  Run on a LOGIN node (needs the Hugging Face login of download_weights.sh):
    source benchmark/env.sh && uv run python benchmark/screen_scenes.py
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from alpasim_utils.artifact import Artifact
from alpasim_utils.scenario import VehicleConfig
from huggingface_hub import HfFileSystem

HF_DATASET = "datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
MEMBERS = ("metadata.yaml", "rig_trajectories.json", "sequence_tracks.json")
OUTPUT = Path("benchmark/scene_screen.csv")
# Vehicles and vulnerable road users; static objects and unknown clutter excluded.
ROAD_USERS = ("automobile", "heavy_truck", "bus", "trailer", "motorcycle",
              "bicycle", "rider", "person", "pedestrian")
LANE_HALF_WIDTH_M = 1.8  # actor centre within this of the ego centre line: same lane
STEP_US = 100_000
MIN_MOVING_SPEED = 2.0  # m/s; headway and TTC are undefined when the ego is stopped
FIELDS = [
    "scene_id", "uuid", "has_lidar", "n_road_users", "n_close_road_users",
    "min_front_gap_m", "min_front_headway_s", "min_front_ttc_s",
    "min_rear_gap_m", "min_rear_ttc_s", "min_side_clearance_m",
    "max_ego_decel_mps2", "ego_mean_speed_mps", "risk",
]


def suite_artifacts(suite: str) -> pl.DataFrame:
    """The artifact the wizard runs for each scene of the suite.

    Matches alpasim_wizard.scenes.sceneset: scenes.scene_ids selects the most
    recently modified artifact per scene_id, which may come from a newer release.
    """
    scenes = pl.read_csv("data/scenes/sim_scenes.csv", infer_schema_length=0)
    suites = pl.read_csv("data/scenes/sim_suites.csv", infer_schema_length=0)
    ids = suites.filter(pl.col("test_suite_id") == suite)["scene_id"].unique()
    return (
        scenes.filter(pl.col("scene_id").is_in(ids))
        .sort("last_modified", descending=True)
        .unique(subset=["scene_id"], keep="first")
        .sort("scene_id")
    )


def fetch_stub(row: dict, workdir: Path) -> Path:
    """Copy the track members of a remote USDZ into a small local zip."""
    url = f"{HF_DATASET}@{row['hf_revision']}/{row['path']}"
    stub = workdir / f"{row['uuid']}.usdz"
    with HfFileSystem().open(url, "rb", block_size=1 << 20) as remote:
        with zipfile.ZipFile(remote) as src, zipfile.ZipFile(stub, "w") as dst:
            for name in MEMBERS:
                dst.writestr(name, src.read(name))
    return stub


def _min(values: list[float]) -> float:
    return min(values, default=float("inf"))


def interaction_metrics(artifact: Artifact) -> dict[str, float]:
    """Closest approaches between the logged ego and the road users around it."""
    vehicle = VehicleConfig()
    ego_front = vehicle.aabb_x_m + vehicle.aabb_x_offset_m  # rig origin to front bumper
    ego_rear = -vehicle.aabb_x_offset_m
    ego_half_width = vehicle.aabb_y_m / 2

    ego = artifact.rig.trajectory
    t0, t1 = ego.time_range_us.start, ego.time_range_us.stop
    times = np.arange(t0, t1, STEP_US)
    ego_xy = np.column_stack([np.interp(times, ego.timestamps_us, ego.positions[:, k]) for k in (0, 1)])
    ego_yaw = np.interp(times, ego.timestamps_us, np.unwrap(ego.yaws))
    ego_speed = np.linalg.norm(np.gradient(ego_xy, STEP_US * 1e-6, axis=0), axis=1)
    heading = np.column_stack([np.cos(ego_yaw), np.sin(ego_yaw)])
    left = np.column_stack([-heading[:, 1], heading[:, 0]])

    front_gap, headway, front_ttc, rear_gap, rear_ttc, side = [], [], [], [], [], []
    road_users = close_road_users = 0
    for obj in artifact.traffic_objects.values():
        if obj.is_static or obj.label_class not in ROAD_USERS:
            continue
        road_users += 1
        ts = obj.trajectory.timestamps_us
        live = (times >= ts[0]) & (times <= ts[-1])
        if live.sum() < 2:
            continue
        xy = np.column_stack([np.interp(times[live], ts, obj.trajectory.positions[:, k]) for k in (0, 1)])
        rel = xy - ego_xy[live]
        lon = np.einsum("ij,ij->i", rel, heading[live])
        lat = np.einsum("ij,ij->i", rel, left[live])
        vel = np.gradient(xy, STEP_US * 1e-6, axis=0)
        actor_fwd = np.einsum("ij,ij->i", vel, heading[live])
        v = ego_speed[live]
        half_len, half_wid = obj.aabb.x / 2, obj.aabb.y / 2
        if np.min(np.hypot(lon, lat)) < 10.0:
            close_road_users += 1

        in_lane = np.abs(lat) < LANE_HALF_WIDTH_M
        ahead = in_lane & (lon > 0)
        gap = lon - ego_front - half_len
        moving = ahead & (v > MIN_MOVING_SPEED) & (gap > 0)
        if ahead.any():
            front_gap.append(float(np.min(gap[ahead])))
        if moving.any():
            headway.append(float(np.min(gap[moving] / v[moving])))
            closing = v - actor_fwd
            approaching = moving & (closing > 0.1)
            if approaching.any():
                front_ttc.append(float(np.min(gap[approaching] / closing[approaching])))

        behind = in_lane & (lon < 0)
        gap_behind = -lon - ego_rear - half_len
        if behind.any():
            rear_gap.append(float(np.min(gap_behind[behind])))
            closing = actor_fwd - v
            catching_up = behind & (closing > 0.1) & (gap_behind > 0)
            if catching_up.any():
                rear_ttc.append(float(np.min(gap_behind[catching_up] / closing[catching_up])))

        alongside = (lon > -ego_rear - half_len) & (lon < ego_front + half_len)
        if alongside.any():
            side.append(float(np.min(np.abs(lat[alongside]) - ego_half_width - half_wid)))

    accel = np.gradient(ego_speed, STEP_US * 1e-6)
    metrics = {
        "n_road_users": road_users,
        "n_close_road_users": close_road_users,
        "min_front_gap_m": _min(front_gap),
        "min_front_headway_s": _min(headway),
        "min_front_ttc_s": _min(front_ttc),
        "min_rear_gap_m": _min(rear_gap),
        "min_rear_ttc_s": _min(rear_ttc),
        "min_side_clearance_m": _min(side),
        "max_ego_decel_mps2": float(max(-np.min(accel), 0.0)),
        "ego_mean_speed_mps": float(np.mean(ego_speed)),
    }
    metrics["risk"] = risk_score(metrics)
    return metrics


def risk_score(m: dict[str, float]) -> float:
    """Sum of how far each margin falls inside a 'tight' threshold, each in [0, 1].

    Thresholds: front headway 1.5 s, front TTC 4 s, rear TTC 3 s, side
    clearance 1 m, ego braking 3 m/s^2 (saturating at 6 m/s^2).
    """
    def inside(value: float, tight: float) -> float:
        return float(np.clip(1.0 - value / tight, 0.0, 1.0))

    return (
        inside(m["min_front_headway_s"], 1.5)
        + inside(m["min_front_ttc_s"], 4.0)
        + inside(m["min_rear_ttc_s"], 3.0)
        + inside(max(m["min_side_clearance_m"], 0.0), 1.0)
        + float(np.clip((m["max_ego_decel_mps2"] - 3.0) / 3.0, 0.0, 1.0))
    )


def screen_one(row: dict) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        artifact = Artifact(source=str(fetch_stub(row, Path(workdir))))
        has_lidar = bool(artifact.metadata.sensors.lidar_ids)
        return {"scene_id": row["scene_id"], "uuid": row["uuid"], "has_lidar": has_lidar,
                **interaction_metrics(artifact)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="public_2601")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="screen only the first N scenes")
    args = parser.parse_args()

    rows = suite_artifacts(args.suite).to_dicts()[: args.limit]
    done = set()
    if OUTPUT.exists():
        done = set(pl.read_csv(OUTPUT)["uuid"].to_list())
    todo = [r for r in rows if r["uuid"] not in done]
    print(f"{len(rows)} scenes in {args.suite}; {len(done)} already screened; {len(todo)} to go")

    write_header = not OUTPUT.exists()
    with OUTPUT.open("a", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        futures = {pool.submit(screen_one, r): r for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # one bad artifact must not stop the screen
                print(f"[{n}/{len(todo)}] {row['scene_id']}: FAILED {exc!r}")
                continue
            writer.writerow(result)
            f.flush()
            print(f"[{n}/{len(todo)}] {row['scene_id']}: risk={result['risk']:.2f}")


if __name__ == "__main__":
    main()
