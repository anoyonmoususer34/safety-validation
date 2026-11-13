"""Scripted hazards injected into finished rollouts, with and without the safety filter.

A CPU-only stress test of the filter, after the car-to-car and pedestrian tests of
emergency-braking ratings.  Each finished rollout gets one hazard at a time:

  lead_brake_*   the vehicle ahead brakes to a stop (alpasim_runtime.lead_brake)
  cut_in_*       a car in the next lane moves into the ego's lane ahead of it,
                 slower than the ego, and may brake once in
  ped_*          a pedestrian crosses the ego's path, in the open or stepping out
                 from behind a van parked at the kerb

The baseline is the logged ego, which never saw the hazard and so drives into it
unless its own plan already slowed: an upper bound on the baseline's crashes.
With the filter, the logged path is driven again with the driver's SafetyFilter in
the loop: at every Drive call it senses the modified scene from where this ego
is (simulated LiDAR is ray-cast against the actor boxes, so the parked van really
hides the pedestrian; radar as in benchmark/replay_validator.py), and the ego
brakes as it commands.

The filter is in the loop, the policy is not: its plans are the logged ones, and
nobody else reacts to the ego.
It tests whether the filter detects each hazard and stops in time.

Writes benchmark/hazard_stress/results.csv and prints a summary.  Run on a LOGIN
node, from the repo root:
    uv run python benchmark/hazard_stress.py
    uv run python benchmark/hazard_stress.py --sources alpamayo_core --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Polars, the Rust log reader and their pools otherwise start a thread per core
# in every worker; login nodes cap the threads per user.
for _pool in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
    os.environ.setdefault(_pool, "1")

import numpy as np
import polars as pl
from alpasim_driver.safety_filter import SafetyFilter, override_decel
from alpasim_driver.schema import SafetyValidatorConfig
from alpasim_grpc.v0 import common_pb2
from alpasim_runtime.config import LeadBrakeConfig
from alpasim_runtime.lead_brake import apply_lead_brake
from alpasim_utils.geometry import Pose, Trajectory, pose_from_grpc
from alpasim_utils.scenario import AABB

sys.path.insert(0, str(Path(__file__).parent))
import counterfactual_braking as cf  # noqa: E402
import replay_validator as rv  # noqa: E402

EGO_ID = "EGO"
HAZARD_ID = "hazard"
OCCLUDER_ID = "hazard_occluder"
SENSORS = ("gt", "radar", "lidar", "fused")
OUTPUT_DIR = Path("benchmark/hazard_stress")
SOURCES = {
    # name: (model, scene set, runs dir, arm)
    "alpamayo_core": ("Alpamayo 1.5", "core", Path("benchmark/runs_core"), "alpamayo1_5"),
    "alpamayo_dev": ("Alpamayo 1.5", "dev", Path("benchmark/runs"), "alpamayo1_5"),
    "simforge_dev": ("SimForge-D1.5", "dev", Path("benchmark/runs"), "simforge_d1_5"),
    "vavam_dev": ("VaVAM", "dev", Path("benchmark/runs"), "vavam"),
}

HAZARD_START_S = 4.0  # after the first Drive call: sensor tracks have settled
MIN_EGO_SPEED_MPS = 5.0
WINDOW_BEFORE_S = 3.0  # the filter runs from this long before the hazard
HAZARD_OVER_S = 10.0  # ... to this long after it starts: walked across or stopped
# Rig height above the road, from the logged vehicles' box bottoms.
EGO_Z_ABOVE_ROAD_M = 1.34
CAR = AABB(4.6, 1.9, 1.5)
PEDESTRIAN = AABB(0.6, 0.6, 1.75)
VAN = AABB(5.0, 2.0, 2.3)
LANE_WIDTH_M = 3.5
KERB_OFFSET_M = LANE_WIDTH_M / 2 + 1.2  # a pedestrian waits this far right of the ego's path


@dataclass(frozen=True)
class Hazard:
    kind: str  # lead_brake, cut_in or pedestrian
    decel_mps2: float = 0.0  # lead_brake; cut_in: braking once in the lane
    gap_m: float = 0.0  # cut_in: bumper gap ahead of the ego when it starts to move over
    slower_mps: float = 0.0  # cut_in: how much slower than the ego
    cut_s: float = 2.0  # cut_in: time to move over
    walk_mps: float = 0.0  # pedestrian
    occluded: bool = False  # pedestrian: steps out from behind a parked van


HAZARDS = {
    "lead_brake_4": Hazard("lead_brake", decel_mps2=4.0),
    "lead_brake_6": Hazard("lead_brake", decel_mps2=6.0),
    "lead_brake_8": Hazard("lead_brake", decel_mps2=8.0),
    "cut_in_10m": Hazard("cut_in", gap_m=10.0, slower_mps=5.0),
    "cut_in_5m": Hazard("cut_in", gap_m=5.0, slower_mps=5.0),
    "cut_in_10m_brake": Hazard("cut_in", gap_m=10.0, slower_mps=3.0, decel_mps2=6.0),
    "ped_walk": Hazard("pedestrian", walk_mps=1.4),
    "ped_run": Hazard("pedestrian", walk_mps=3.0),
    "ped_walk_occluded": Hazard("pedestrian", walk_mps=1.4, occluded=True),
    "ped_run_occluded": Hazard("pedestrian", walk_mps=3.0, occluded=True),
}


def _pose(xy: np.ndarray, yaw: float, z: float) -> common_pb2.Pose:
    pose = common_pb2.Pose()
    pose.vec.x, pose.vec.y, pose.vec.z = float(xy[0]), float(xy[1]), z
    pose.quat.w, pose.quat.z = math.cos(yaw / 2), math.sin(yaw / 2)
    return pose


def _hazard_time(log: rv.RolloutLog, path: cf.LoggedPath) -> int | None:
    """The first logged time, HAZARD_START_S into the drive, the ego is moving."""
    earliest = log.drives[0].time_us + int(HAZARD_START_S * 1e6)
    latest = path.times[-1] - int(3e6)
    for t in path.times:
        if earliest <= t <= latest and path.speed_at(int(t)) >= MIN_EGO_SPEED_MPS:
            return int(t)
    return None


def _add_actor(log, actor_id, aabb, label, poses) -> None:
    log.aabbs[actor_id] = aabb
    log.labels[actor_id] = label
    log.actor_poses[actor_id] = poses


def _road_z(path: cf.LoggedPath, s: float, aabb: AABB) -> float:
    return path.at(s)[2] - EGO_Z_ABOVE_ROAD_M + aabb.z / 2


def _lead_brake(log, path, hazard) -> tuple[str, int] | None:
    ego = rv._trajectory(log.actor_poses[EGO_ID])
    found = apply_lead_brake(
        rv._traffic(log),
        ego,
        log.drives[0].time_us + int(HAZARD_START_S * 1e6),
        int(path.times[-1]),
        replace(LeadBrakeConfig(), decel_mps2=hazard.decel_mps2),
    )
    if found is None:
        return None
    traffic, lead, brake_us = found
    braked = traffic[lead].trajectory
    poses = dict(log.actor_poses[lead])
    for t in path.times:
        if t >= brake_us and t in braked.time_range_us:
            p = braked.interpolate_pose(int(t))
            poses[int(t)] = _pose(p.vec3[:2], p.yaw(), float(p.vec3[2]))
    log.actor_poses[lead] = poses
    return lead, brake_us


def _cut_in(log, path, hazard) -> tuple[str, int] | None:
    start_us = _hazard_time(log, path)
    if start_us is None:
        return None
    ego_length = log.aabbs[EGO_ID].x
    speed = path.speed_at(start_us) - hazard.slower_mps
    if speed <= 0.0:
        return None
    s0 = path.s_at(start_us) + ego_length / 2 + hazard.gap_m + CAR.x / 2
    in_lane_us = start_us + hazard.cut_s * 1e6
    poses = {}
    for t in path.times:
        dt = (t - start_us) * 1e-6
        if hazard.decel_mps2 > 0.0 and t > in_lane_us:
            braking = min((t - in_lane_us) * 1e-6, speed / hazard.decel_mps2)
            s = s0 + speed * hazard.cut_s + speed * braking - 0.5 * hazard.decel_mps2 * braking**2
            v = speed - hazard.decel_mps2 * braking
        else:
            s, v = s0 + speed * dt, speed
        # Smoothstep from the next lane (left) to the ego's.
        u = float(np.clip(dt / hazard.cut_s, 0.0, 1.0))
        lateral = LANE_WIDTH_M * (1.0 - u * u * (3.0 - 2.0 * u))
        lateral_speed = -LANE_WIDTH_M * 6.0 * u * (1.0 - u) / hazard.cut_s
        xy, yaw, _ = path.at(s)
        normal = np.array([-math.sin(yaw), math.cos(yaw)])
        heading = yaw + math.atan2(lateral_speed, max(v, 0.1))
        poses[int(t)] = _pose(xy + lateral * normal, heading, _road_z(path, s, CAR))
    _add_actor(log, HAZARD_ID, CAR, "automobile", poses)
    return HAZARD_ID, start_us


def _pedestrian(log, path, hazard) -> tuple[str, int] | None:
    """A pedestrian from the right kerb, timed to meet the ego's front bumper."""
    meet_us = _hazard_time(log, path)
    if meet_us is None:
        return None
    meet_us += int(2e6)  # leave the filter time to see them coming
    if meet_us > path.times[-1]:
        return None
    s_cross = path.s_at(meet_us) + log.aabbs[EGO_ID].x / 2 + PEDESTRIAN.x / 2
    xy_cross, yaw, _ = path.at(s_cross)
    normal = np.array([-math.sin(yaw), math.cos(yaw)])
    kerb = -KERB_OFFSET_M  # right of the lane, behind where a van parks
    # Stand at the kerb until it is time to walk out, then cross at a constant pace
    # to beyond the far side.
    walk_start_us = meet_us + kerb / hazard.walk_mps * 1e6
    poses = {}
    for t in path.times:
        walked = max(t - walk_start_us, 0.0) * 1e-6 * hazard.walk_mps
        lateral = min(kerb + walked, -kerb + 2.0)
        poses[int(t)] = _pose(
            xy_cross + lateral * normal, yaw + math.pi / 2, _road_z(path, s_cross, PEDESTRIAN)
        )
    _add_actor(log, HAZARD_ID, PEDESTRIAN, "person", poses)
    if hazard.occluded:
        s_van = s_cross - VAN.x / 2 - 1.0
        van_xy, van_yaw, _ = path.at(s_van)
        van_normal = np.array([-math.sin(van_yaw), math.cos(van_yaw)])
        van = _pose(
            van_xy + (kerb + 0.3) * van_normal, van_yaw, _road_z(path, s_van, VAN)
        )
        _add_actor(log, OCCLUDER_ID, VAN, "automobile", {int(t): van for t in path.times})
    return HAZARD_ID, meet_us


def hazard_onset_us(hazard: Hazard, start_us: int) -> int:
    """When the hazard starts, for inject's ``start_us``: the brake or the move
    over, or the pedestrian stepping off the kerb (start_us is when they meet the path)."""
    if hazard.kind == "pedestrian":
        return int(start_us - KERB_OFFSET_M / hazard.walk_mps * 1e6)
    return start_us


def _impact_speed(log: rv.RolloutLog, hazard_id: str, track: list, hit_us: int | None) -> float | None:
    """Speed of the ego relative to the hazard along the ego's heading at ``hit_us``."""
    if hit_us is None:
        return None
    t, _, yaw, ego_v, _ = min(track, key=lambda step: abs(step[0] - hit_us))
    hazard = cf.LoggedPath(log.actor_poses[hazard_id])
    k = int(np.clip(np.searchsorted(hazard.times, t), 1, len(hazard.times) - 1))
    velocity = (hazard.xy[k] - hazard.xy[k - 1]) / ((hazard.times[k] - hazard.times[k - 1]) * 1e-6)
    return float(ego_v - velocity @ np.array([math.cos(yaw), math.sin(yaw)]))


def inject(log: rv.RolloutLog, hazard: Hazard) -> tuple[rv.RolloutLog, str, int] | None:
    """A copy of ``log`` with ``hazard``, its actor ID and when it starts."""
    log = copy.deepcopy(log)
    path = cf.LoggedPath(log.actor_poses[EGO_ID])
    build = {"lead_brake": _lead_brake, "cut_in": _cut_in, "pedestrian": _pedestrian}
    found = build[hazard.kind](log, path, hazard)
    return None if found is None else (log, *found)


def _only(log: rv.RolloutLog, keep: set[str]) -> rv.RolloutLog:
    """``log`` with only the ego and the actors in ``keep``."""
    out = copy.copy(log)
    out.actor_poses = {a: p for a, p in log.actor_poses.items() if a == EGO_ID or a in keep}
    return out


def _collision(result: dict, t0: int) -> tuple[float | None, str | None, bool | None]:
    if result["collision_us"] is None:
        return None, None, None
    return (result["collision_us"] - t0) * 1e-6, result["collision_actor"], result["collision_from_behind"]


def _se3(xy: np.ndarray, yaw: float, z: float) -> np.ndarray:
    se3 = np.eye(4)
    se3[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    se3[:2, 3], se3[2, 3] = xy, z
    return se3


def closed_loop(
    log: rv.RolloutLog,
    hazard_id: str,
    start_us: int,
    sensor: str,
    radars: list,
    cfg: SafetyValidatorConfig,
    single_candidate: bool,
    lidar_dropout: float = 0.0,
) -> dict:
    """Drive the logged ego's path again with the filter in the loop: from
    WINDOW_BEFORE_S before the hazard on, the filter senses the scene from where
    this ego is, at its speed, and slows it as it commands; the policy's plans are
    the logged ones.  The first collision with the hazard and with anyone else."""
    ego = log.actor_poses[EGO_ID]
    times = np.array(sorted(ego))
    path = cf.LoggedPath(ego)
    dt = np.diff(times) * 1e-6
    logged_speed = np.diff(path.arc) / dt
    size = log.aabbs[EGO_ID]
    # The logged rig sits a fixed distance behind the ego box's centre.
    t_ref = next(t for t in times if t in log.ego_poses)
    rig_ref = pose_from_grpc(log.ego_poses[t_ref]).as_se3()
    box_ref = pose_from_grpc(ego[t_ref]).as_se3()
    rig_behind_m = float((box_ref[:2, 3] - rig_ref[:2, 3]) @ rig_ref[:2, 0])
    rig_below_m = float(box_ref[2, 3] - rig_ref[2, 3])

    def rig_se3(s: float) -> np.ndarray:
        xy, yaw, z = path.at(s)
        return _se3(xy - rig_behind_m * np.array([math.cos(yaw), math.sin(yaw)]), yaw, z - rig_below_m)

    # Each Drive call acts from the first actor-pose time at or after it.
    drives = {}
    for drive in log.drives:
        t = drive.selected.poses[0].timestamp_us
        if t >= start_us - WINDOW_BEFORE_S * 1e6 and t in log.ego_speed:
            k = int(np.searchsorted(times, t))
            if k < len(times):
                drives.setdefault(int(times[k]), drive)

    safety_filter = SafetyFilter(cfg, rv.OUTPUT_FREQUENCY_HZ)
    traffic = rv._traffic(log)
    s, v = 0.0, logged_speed[0]
    command = None
    overrides, first_override_us = 0, None
    # At each Drive call: the rig pose, the obstacles sensed (rig frame) and the decision.
    decisions = []
    # (time, position, heading, speed, logged speed) of this ego at each step.
    track = [(int(times[0]), path.xy[0], float(path.yaw[0]), v, logged_speed[0])]
    hazard_hit_us = None
    for k in range(len(times) - 1):
        t = int(times[k])
        if t > start_us + HAZARD_OVER_S * 1e6:
            break
        drive = drives.get(t)
        if drive is not None:
            se3 = rig_se3(s)
            prev_se3 = rig_se3(max(s - v * rv.VELOCITY_DT_US * 1e-6, 0.0))
            ego_trajectory = Trajectory.from_poses(
                timestamps=np.array([t - rv.VELOCITY_DT_US, t], dtype=np.uint64),
                poses=[Pose.from_se3(m.astype(np.float32)) for m in (prev_se3, se3)],
            )
            logged_v = log.ego_speed[drive.selected.poses[0].timestamp_us]
            yaw_rate = log.ego_yaw_rate[drive.selected.poses[0].timestamp_us] * v / max(logged_v, 0.1)
            sensed = rv.sense(
                safety_filter, sensor, traffic, ego_trajectory, t, np.linalg.inv(se3),
                v, yaw_rate, radars, seed=0,
                mode=None if sensor == "gt" else rv.FIXED_MODES[sensor],
                lidar_dropout=lidar_dropout,
            )
            logged_rig = pose_from_grpc(drive.selected.poses[0].pose).as_se3().astype(np.float64)
            prediction = rv._prediction(drive, np.linalg.inv(logged_rig), single_candidate)
            _, debug = safety_filter.apply(prediction, sensed, v)
            decisions.append({
                "time_us": t, "rig_se3": se3, "speed": v,
                "obstacles": [] if sensed is None else sensed.obstacles,
                "reason": debug["reason"], "safe_speed": debug.get("safe_speed"),
                "violations": list(debug["violations"]),
            })
            command = None
            if debug["reason"] == "speed_override":
                emergency = "EMERGENCY_TTC_VIOLATION" in debug["violations"]
                safe = debug["safe_speed"]
                command = (safe, override_decel(cfg, v, safe, emergency))
                overrides += 1
                first_override_us = first_override_us or t
        if command is None:
            # Behind the logged drive, catch up at ACCEL_MPS2; on it, follow it exactly.
            lagging = s < path.arc[k] - 1e-6
            v_next = min(logged_speed[k], v + cf.ACCEL_MPS2 * dt[k]) if lagging else logged_speed[k]
        else:
            safe, decel = command
            v_next = max(min(safe, logged_speed[k]), v - decel * dt[k])
            v_next = min(v_next, v + cf.ACCEL_MPS2 * dt[k])
        s = min(s + 0.5 * (v + v_next) * dt[k], path.arc[k + 1])
        v = v_next
        xy, yaw, _ = path.at(s)
        track.append((int(times[k + 1]), xy, yaw, v, logged_speed[min(k + 1, len(logged_speed) - 1)]))
        hazard_box = dict(cf._actor_boxes(log, int(times[k + 1]))).get(hazard_id)
        if hazard_box is not None and cf._boxes_overlap(
            cf._box_corners(xy, yaw, size.x, size.y), hazard_box
        ):
            hazard_hit_us = int(times[k + 1])
            break
    return {
        "hazard_hit_us": hazard_hit_us,
        "track": track,
        "decisions": decisions,
        "overrides": overrides,
        "first_override_us": first_override_us,
    }


def _one(job: dict) -> list[dict]:
    log = asyncio.run(rv._read_log(job["asl_path"]))
    injected = inject(log, HAZARDS[job["hazard"]])
    base = {
        "source": job["source"], "model": job["model"], "scenes": job["scenes"],
        "scene": job["asl_path"].parents[1].name, "rollout": job["asl_path"].parent.name,
        "hazard": job["hazard"],
    }
    if injected is None:
        return [{**base, "applicable": False}]
    mod, hazard_id, start_us = injected
    t0 = min(mod.actor_poses[EGO_ID])
    hazard_only = _only(mod, {hazard_id})
    logged_hazard = cf.drive_counterfactual(hazard_only, {})
    cfg = job["cfg"]
    hit_t, _, _ = _collision(logged_hazard, t0)
    onset_us = hazard_onset_us(HAZARDS[job["hazard"]], start_us)
    # The logged ego's first collision with anyone but the hazard, per follower model.
    logged_track, _ = cf.counterfactual_track(mod, {})
    logged_other = {"": cf.first_collision(mod, hazard_id, logged_track, {})}
    for reaction_s in cf.REACTION_TIMES_S:
        logged_other[f"_react_{reaction_s:g}s"] = cf.first_collision(
            mod, hazard_id, logged_track, cf.reacting_followers(mod, hazard_id, logged_track, reaction_s)
        )
    logged_other = {tag: None if c is None else c[1] for tag, c in logged_other.items()}
    rows = []
    for sensor in job["sensors"]:
        result = closed_loop(
            mod, hazard_id, start_us, sensor, job["radars"], cfg, job["single_candidate"],
            job["lidar_dropout"],
        )
        others = {"": cf.first_collision(mod, hazard_id, result["track"], {})}
        for reaction_s in cf.REACTION_TIMES_S:
            followers = cf.reacting_followers(mod, hazard_id, result["track"], reaction_s)
            others[f"_react_{reaction_s:g}s"] = cf.first_collision(mod, hazard_id, result["track"], followers)
        new_other = {
            tag: other is not None and other[1] != logged_other[tag] for tag, other in others.items()
        }
        # The validator's slowdowns from the hazard's onset on.
        reacting = [d for d in result["decisions"] if d["reason"] == "speed_override" and d["time_us"] >= onset_us]
        decels = [
            override_decel(cfg, d["speed"], d["safe_speed"], "EMERGENCY_TTC_VIOLATION" in d["violations"])
            for d in reacting
        ]
        rows.append({
            **base,
            "applicable": True,
            "sensor": sensor,
            "hazard_start_s": (start_us - t0) * 1e-6,
            "ego_speed_at_start_mps": cf.LoggedPath(mod.actor_poses[EGO_ID]).speed_at(start_us),
            "baseline_hits_hazard": hit_t is not None,
            "baseline_hit_s": hit_t,
            "filtered_hits_hazard": result["hazard_hit_us"] is not None,
            "filtered_hit_s": (
                None if result["hazard_hit_us"] is None else (result["hazard_hit_us"] - t0) * 1e-6
            ),
            "first_override_s": (
                None if result["first_override_us"] is None
                else (result["first_override_us"] - t0) * 1e-6
            ),
            "overrides": result["overrides"],
            "hazard_onset_s": (onset_us - t0) * 1e-6,
            "first_reaction_s": (reacting[0]["time_us"] - t0) * 1e-6 if reacting else None,
            "first_reaction_constraints": "|".join(reacting[0]["violations"]) if reacting else None,
            "commanded_decels": "|".join(f"{d:.2f}" for d in decels),
            "baseline_impact_speed_mps": _impact_speed(mod, hazard_id, logged_track, logged_hazard["collision_us"]),
            "filtered_impact_speed_mps": _impact_speed(mod, hazard_id, result["track"], result["hazard_hit_us"]),
            **{f"new_other_collision{tag}": new for tag, new in new_other.items()},
            **{
                f"new_other_from_behind{tag}": others[tag][2] if new_other[tag] else None
                for tag in others
            },
        })
    return rows


def _asl_paths(runs_dir: Path, arm: str) -> list[Path]:
    return sorted(
        p for d in runs_dir.glob(f"{arm}*")
        if d.name == arm or d.name.startswith(f"{arm}__p")
        for p in d.glob("rollouts/*/*/rollout.asl")
        if (d / "aggregate" / "results-summary.json").exists()
    )


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    applied = results.filter(pl.col("applicable"))
    return (
        applied.with_columns(
            pl.col("hazard").str.replace(r"_.*", "").alias("family"),
            (pl.col("baseline_hits_hazard") & ~pl.col("filtered_hits_hazard")).alias("prevented"),
            (pl.col("new_other_collision") & ~pl.col("new_other_from_behind").fill_null(False)).alias("new_front_side"),
        )
        .group_by("model", "hazard", "sensor")
        .agg(
            pl.len().alias("rollouts"),
            pl.col("baseline_hits_hazard").sum().alias("baseline_crashes"),
            pl.col("filtered_hits_hazard").sum().alias("crashes_with_filter"),
            pl.col("prevented").sum().alias("prevented"),
            pl.col("new_front_side").sum().alias("new_front_side"),
        )
        .sort("model", "hazard", "sensor")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    parser.add_argument("--hazards", nargs="+", default=list(HAZARDS), choices=HAZARDS)
    parser.add_argument("--limit", type=int, help="rollouts per source")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "results.csv")
    parser.add_argument("--sensors", nargs="+", default=list(SENSORS), choices=SENSORS)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="a SafetyValidatorConfig setting in place of its default, e.g. crossing_horizon_s=0",
    )
    parser.add_argument(
        "--single-candidate", action="store_true",
        help="give the filter only the policy's own plan, as a one-sample policy would",
    )
    parser.add_argument(
        "--radar", action="append", default=[], metavar="KEY=VALUE",
        help="a RadarConfig setting for every radar, e.g. false_alarms_per_scan=3",
    )
    parser.add_argument(
        "--lidar-dropout", type=float, default=0.0, help="fraction of LiDAR returns lost at random",
    )
    args = parser.parse_args()

    overrides = {key: float(value) for key, value in (item.split("=", 1) for item in args.radar)}
    radars = [replace(radar, **overrides) for radar in rv._load_radars(rv.DEFAULT_RADAR_CONFIG)]
    jobs = [
        {"source": source, "model": model, "scenes": scenes, "asl_path": path,
         "hazard": hazard, "radars": radars, "cfg": rv.validator_config(args.set),
         "single_candidate": args.single_candidate, "sensors": args.sensors,
         "lidar_dropout": args.lidar_dropout}
        for source in args.sources
        for model, scenes, runs_dir, arm in [SOURCES[source]]
        for path in _asl_paths(runs_dir, arm)[: args.limit]
        for hazard in args.hazards
    ]
    print(f"{len(jobs)} rollout x hazard jobs")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, job_rows in enumerate(pool.map(_one, jobs), 1):
            rows.extend(job_rows)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
                pl.DataFrame(rows, infer_schema_length=None).write_csv(args.output)
    results = pl.DataFrame(rows, infer_schema_length=None)
    results.write_csv(args.output)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(summarize(results))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
