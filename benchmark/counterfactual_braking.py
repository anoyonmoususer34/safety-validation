"""Estimate which logged collisions the safety filter's slowdowns would have avoided.

For every rollout replayed by benchmark/replay_validator.py, this drives the logged
ego again along its own logged path, but at the speed the filter commanded: at
each Drive call the filter overrode, the ego brakes toward that call's safe speed
at the filter's own rate (safety_filter.override_decel), and otherwise it follows
the logged speed again (at most ACCEL_MPS2 faster).  Every other actor replays
its logged trajectory, and the ego's box is checked against theirs at each step.

An estimate, not a closed-loop result: the filter's decisions are the ones it took
on the logged scene, the ego never steers away, and nobody reacts to it.  In a
rollout that crashed, the filter's last slowdown in the second before the
logged collision holds from then on: by then the filter sees the scene from an
ego that is crashing, and loses sight of what it hits.  A following car that would brake for a slowing ego drives into it here, so new
rear collisions are an upper bound.

Writes RUNS_DIR/REPLAY_NAME/counterfactual_SENSOR.csv and prints a summary.
Run on a LOGIN node, from the repo root, after replay_validator.py:
    uv run python benchmark/counterfactual_braking.py benchmark/runs_core validator_replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Polars, the Rust log reader and their pools otherwise start a thread per core
# in every worker; login nodes cap the threads per user.
for _pool in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
    os.environ.setdefault(_pool, "1")

import numpy as np
import polars as pl
from alpasim_driver.safety_filter import override_decel
from alpasim_driver.schema import SafetyValidatorConfig
from alpasim_utils.geometry import pose_from_grpc

sys.path.insert(0, str(Path(__file__).parent))
import replay_validator as rv  # noqa: E402

EGO_ID = "EGO"
ACCEL_MPS2 = 2.0  # how fast the ego regains the logged speed after an override
FREEZE_WINDOW_US = 1_000_000
# Cars behind the ego that brake for it (reacting_followers).
REACTION_TIMES_S = (1.0, 1.5)
FOLLOWER_DECEL_MPS2 = 6.0
FOLLOWER_TTC_S = 3.0  # it brakes once this close to running into the ego ...
FOLLOWER_MIN_GAP_M = 2.0  # ... or once this close behind it
FOLLOWER_BEHIND_M = 50.0
FOLLOWER_LATERAL_M = 2.5
FOLLOWER_HEADING_RAD = math.radians(45.0)
CATCH_UP_US = 1_000_000


def _box_corners(xy: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    return xy + half @ np.array([[c, s], [-s, c]])


def _boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two (4, 2) rectangles."""
    for box in (a, b):
        for i in range(2):
            axis = box[i + 1] - box[i]
            axis = np.array([-axis[1], axis[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _yaw(se3: np.ndarray) -> float:
    return float(np.arctan2(se3[1, 0], se3[0, 0]))


def _actor_boxes(log: rv.RolloutLog, time_us: int) -> list[tuple[str, np.ndarray]]:
    boxes = []
    for actor_id, poses in log.actor_poses.items():
        if actor_id == EGO_ID or time_us not in poses:
            continue
        se3 = pose_from_grpc(poses[time_us]).as_se3()
        size = log.aabbs[actor_id]
        boxes.append((actor_id, _box_corners(se3[:2, 3], _yaw(se3), size.x, size.y)))
    return boxes


def _commanded_speeds(
    steps: pl.DataFrame, cfg: SafetyValidatorConfig
) -> dict[int, tuple[float, float] | None]:
    """Drive-call time -> (safe speed, braking) where the filter slowed the plan, None
    where it did not.  A command holds until the next Drive call."""
    emergency = steps["violations"].fill_null("").str.contains("EMERGENCY_TTC")
    return {
        t: (v, override_decel(cfg, speed, v, e)) if r == "speed_override" else None
        for t, r, v, e, speed in zip(
            steps["time_us"], steps["reason"], steps["safe_speed_mps"], emergency, steps["speed_mps"]
        )
    }


def _freeze_time(commanded: dict[int, tuple[float, float] | None], collision_us: int | None) -> int | None:
    """The decision that holds from a logged collision on: the last slowdown in the
    FREEZE_WINDOW_US before it, else the last decision before it."""
    if collision_us is None:
        return None
    recent = [
        t for t, c in commanded.items()
        if c is not None and collision_us - FREEZE_WINDOW_US <= t <= collision_us
    ]
    return max(recent, default=collision_us)


def _came_from_behind(
    log: rv.RolloutLog, actor_id: str, history: list, boxes: dict[int, np.ndarray]
) -> bool:
    """Whether the actor was behind the re-driven ego CATCH_UP_US before hitting it,
    at its place in ``boxes`` (a reacting follower's) or else its logged one."""
    t, position, heading = min(history, key=lambda h: abs(h[0] - (history[-1][0] - CATCH_UP_US)))
    poses = log.actor_poses[actor_id]
    if t in boxes:
        actor_xy = boxes[t].mean(axis=0)
    elif t in poses:
        actor_xy = pose_from_grpc(poses[t]).as_se3()[:2, 3]
    else:
        return False
    return float((actor_xy - position) @ np.array([np.cos(heading), np.sin(heading)])) < 0.0


class LoggedPath:
    """A logged path by distance travelled, extended straight at both ends."""

    def __init__(self, poses: dict[int, common_pb2.Pose]) -> None:
        self.times = np.array(sorted(poses))
        se3s = [pose_from_grpc(poses[t]).as_se3() for t in self.times]
        self.xy = np.array([m[:2, 3] for m in se3s], dtype=np.float64)
        self.z = np.array([m[2, 3] for m in se3s], dtype=np.float64)
        self.yaw = np.unwrap([math.atan2(m[1, 0], m[0, 0]) for m in se3s])
        self.arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        )

    def s_at(self, time_us: float) -> float:
        return float(np.interp(time_us, self.times, self.arc))

    def speed_at(self, time_us: int) -> float:
        k = int(np.clip(np.searchsorted(self.times, time_us), 1, len(self.times) - 1))
        return float(
            (self.arc[k] - self.arc[k - 1]) / ((self.times[k] - self.times[k - 1]) * 1e-6)
        )

    def at(self, s: float) -> tuple[np.ndarray, float, float]:
        """Position, heading and rig height ``s`` metres along the path."""
        end = s - self.arc[-1] if s > self.arc[-1] else min(s, 0.0)
        s_in = float(np.clip(s, 0.0, self.arc[-1]))
        yaw = float(np.interp(s_in, self.arc, self.yaw))
        xy = np.array([np.interp(s_in, self.arc, self.xy[:, i]) for i in range(2)])
        xy = xy + end * np.array([math.cos(yaw), math.sin(yaw)])
        return xy, yaw, float(np.interp(s_in, self.arc, self.z))


def reacting_followers(
    log: rv.RolloutLog, hazard_id: str | None, track: list, reaction_s: float
) -> dict[str, dict[int, np.ndarray]]:
    """Time -> box corners of each car behind the ego that brakes for it.

    A car up to FOLLOWER_BEHIND_M behind the ego in its lane, heading the same way,
    drives its recording until it is closing on the ego with under FOLLOWER_TTC_S
    to contact, or within FOLLOWER_MIN_GAP_M of it.  ``reaction_s`` later it brakes
    at up to FOLLOWER_DECEL_MPS2 to the ego's speed, never faster than recorded.
    The rule is the same whether or not the filter slows the ego."""
    ego_length = log.aabbs[EGO_ID].x
    ego_xy = np.array([xy for _, xy, _, _, _ in track])
    followers = {}
    for actor_id, poses in log.actor_poses.items():
        if actor_id in (EGO_ID, hazard_id) or len(poses) < 2:
            continue
        path = LoggedPath(poses)
        nearest = np.min(np.linalg.norm(path.xy[:, None, :] - ego_xy[None, ::5, :], axis=2))
        if nearest > FOLLOWER_BEHIND_M + 10.0:
            continue
        size = log.aabbs[actor_id]
        s = v = prev_t = react_us = None
        boxes = {}
        for t, xy_ego, yaw_ego, v_ego, _ in track:
            if s is None and not path.times[0] <= t <= path.times[-1]:
                continue
            if s is None or react_us is None or t < react_us:
                s, v = path.s_at(t), path.speed_at(t)
            else:
                dt = (t - prev_t) * 1e-6
                target = min(path.speed_at(t), v_ego)
                v_next = max(target, v - FOLLOWER_DECEL_MPS2 * dt) if target < v else min(
                    target, v + ACCEL_MPS2 * dt
                )
                s += 0.5 * (v + v_next) * dt
                v = v_next
            prev_t = t
            xy, yaw, _ = path.at(s)
            if react_us is None:
                c, si = math.cos(yaw_ego), math.sin(yaw_ego)
                x, y = c * (xy[0] - xy_ego[0]) + si * (xy[1] - xy_ego[1]), -si * (xy[0] - xy_ego[0]) + c * (xy[1] - xy_ego[1])
                heading = (yaw - yaw_ego + math.pi) % (2 * math.pi) - math.pi
                if -FOLLOWER_BEHIND_M <= x < 0.0 and abs(y) <= FOLLOWER_LATERAL_M and abs(heading) <= FOLLOWER_HEADING_RAD:
                    gap = -x - (ego_length + size.x) / 2
                    closing = v - v_ego
                    if gap <= FOLLOWER_MIN_GAP_M or (closing > 0.0 and gap / closing < FOLLOWER_TTC_S):
                        react_us = t + reaction_s * 1e6
            if react_us is not None:
                boxes[t] = _box_corners(xy, yaw, size.x, size.y)
        if boxes:
            followers[actor_id] = boxes
    return followers


def first_collision(
    log: rv.RolloutLog, hazard_id: str | None, track: list, followers: dict[str, dict[int, np.ndarray]]
) -> tuple[int, str, bool] | None:
    """The ego's first collision along ``track`` with anyone but ``hazard_id``, the
    ``followers`` at their reacting places: time, actor, whether from behind."""
    size = log.aabbs[EGO_ID]
    history = []
    for t, xy, yaw, _, _ in track:
        history.append((t, xy, yaw))
        corners = _box_corners(xy, yaw, size.x, size.y)
        for actor_id, box in _actor_boxes(log, t):
            if actor_id == hazard_id:
                continue
            follower = followers.get(actor_id, {})
            box = follower.get(t, box)
            if _boxes_overlap(corners, box):
                return t, actor_id, _came_from_behind(log, actor_id, history, follower)
    return None


def counterfactual_track(
    log: rv.RolloutLog,
    commanded: dict[int, tuple[float, float] | None],
    freeze_us: int | None = None,
) -> tuple[list, float]:
    """Re-drive the ego's logged path at the commanded speeds: its (time, position,
    heading, speed, logged speed) at each step, and the distance it fell short.

    Decisions after ``freeze_us`` are ignored and the last one before it holds:
    the filter saw the scene from the logged ego, which has crashed by then."""
    path = LoggedPath(log.actor_poses[EGO_ID])
    times = path.times
    dt = np.diff(times) * 1e-6
    logged_speed = np.diff(path.arc) / dt
    decision_times = np.array(sorted(t for t in commanded if freeze_us is None or t <= freeze_us))

    s, v = 0.0, logged_speed[0]
    track = [(int(times[0]), path.xy[0], float(path.yaw[0]), v, logged_speed[0])]
    for k in range(len(times) - 1):
        recent = decision_times[decision_times <= times[k]]
        override = commanded[recent[-1]] if len(recent) else None
        if override is None:
            # Behind the logged drive, catch up at ACCEL_MPS2; on it, follow it exactly.
            lagging = s < path.arc[k] - 1e-6
            v_next = min(logged_speed[k], v + ACCEL_MPS2 * dt[k]) if lagging else logged_speed[k]
        else:
            safe_speed, decel = override
            v_next = max(min(safe_speed, logged_speed[k]), v - decel * dt[k])
            v_next = min(v_next, v + ACCEL_MPS2 * dt[k])
        s = min(s + 0.5 * (v + v_next) * dt[k], path.arc[k + 1])
        v = v_next
        xy, yaw, _ = path.at(s)
        track.append((int(times[k + 1]), xy, yaw, v, logged_speed[min(k + 1, len(logged_speed) - 1)]))
    return track, float(path.arc[-1] - s)


def drive_counterfactual(
    log: rv.RolloutLog,
    commanded: dict[int, tuple[float, float] | None],
    freeze_us: int | None = None,
) -> dict:
    """Re-drive the ego's logged path at the commanded speeds; first collision with
    the logged traffic and progress (see counterfactual_track)."""
    track, shortfall = counterfactual_track(log, commanded, freeze_us)
    first = first_collision(log, None, track, {})
    return {
        "collision_us": first[0] if first else None,
        "collision_actor": first[1] if first else None,
        "collision_from_behind": first[2] if first else None,
        "progress_lost_m": shortfall if first is None else None,
    }


def _outcome(logged: tuple | None, filtered: tuple | None) -> str:
    if logged is None and filtered is None:
        return "no collision either way"
    if filtered is None:
        return "avoided"
    if logged is None:
        return "new collision"
    if filtered[1] == logged[1]:
        return "same actor"
    return "only hit from behind" if filtered[2] else "different actor"


def _one(job: dict) -> dict:
    log = asyncio.run(rv._read_log(job["asl_path"]))
    t0 = min(log.actor_poses[EGO_ID])
    logged_track, _ = counterfactual_track(log, {})
    logged = {"": first_collision(log, None, logged_track, {})}
    track, shortfall = counterfactual_track(
        log, job["commanded"], _freeze_time(job["commanded"], logged[""][0] if logged[""] else None)
    )
    filtered = {"": first_collision(log, None, track, {})}
    for reaction_s in REACTION_TIMES_S:
        tag = f"_react_{reaction_s:g}s"
        logged[tag] = first_collision(
            log, None, logged_track, reacting_followers(log, None, logged_track, reaction_s)
        )
        filtered[tag] = first_collision(
            log, None, track, reacting_followers(log, None, track, reaction_s)
        )
    row = {
        "scene": job["scene"],
        "rollout": job["rollout"],
        "at_fault": job["at_fault"],
        "eval_collision": job["collision"],
        "overrides": sum(v is not None for v in job["commanded"].values()),
        "progress_lost_m": shortfall if filtered[""] is None else None,
    }
    # Without a tag, the logged traffic as recorded; with one, cars behind react.
    # The untagged box test on the logged drive should find the eval's collision.
    for tag in filtered:
        for name, collision in (("logged", logged[tag]), ("filtered", filtered[tag])):
            row[f"{name}_collision_s{tag}"] = None if collision is None else (collision[0] - t0) * 1e-6
            row[f"{name}_actor{tag}"] = collision[1] if collision else None
            row[f"{name}_from_behind{tag}"] = collision[2] if collision else None
        row[f"outcome{tag}"] = _outcome(logged[tag], filtered[tag])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("replay_name", help="replay_validator.py --output-name")
    parser.add_argument("--sensors", nargs="+", default=["gt", "radar", "lidar", "fused"])
    parser.add_argument("--arm", default="alpamayo1_5")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    replay_dir = args.runs_dir / args.replay_name
    cfg = SafetyValidatorConfig(enabled=True)
    asl_paths = {
        p.parent.name: p
        for p in args.runs_dir.glob(f"{args.arm}*/rollouts/*/*/rollout.asl")
        if (p.parents[3] / "aggregate" / "results-summary.json").exists()
    }
    for sensor in args.sensors:
        rollouts = pl.read_csv(replay_dir / f"rollouts_{sensor}.csv")
        steps = pl.read_csv(replay_dir / f"steps_{sensor}.csv", infer_schema_length=None).with_columns(
            pl.col("safe_speed_mps").cast(pl.Float64, strict=False)
        )
        jobs = [
            {
                "scene": r["scene"],
                "rollout": r["rollout"],
                "at_fault": r["at_fault"],
                "collision": r["collision"],
                "asl_path": asl_paths[r["rollout"]],
                "commanded": _commanded_speeds(steps.filter(pl.col("rollout") == r["rollout"]), cfg),
            }
            for r in rollouts.to_dicts()
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            result = pl.DataFrame(list(pool.map(_one, jobs)), infer_schema_length=None)
        result.write_csv(replay_dir / f"counterfactual_{sensor}.csv")
        crashed = result.filter(pl.col("eval_collision").is_not_null())
        clean = result.filter(pl.col("eval_collision").is_null())
        print(f"== {sensor}")
        print(json.dumps({
            "logged crashes found by the box test": f"{crashed['logged_collision_s'].is_not_null().sum()}/{crashed.height}",
            "logged crashes avoided": f"{(crashed['outcome'] == 'avoided').sum()}/{crashed.height}",
            "at-fault crashes avoided": f"{crashed.filter(pl.col('at_fault'))['outcome'].eq('avoided').sum()}/{crashed['at_fault'].sum()}",
            "at-fault crashes avoided, then hit from behind by non-reacting traffic": f"{crashed.filter(pl.col('at_fault'))['outcome'].eq('only hit from behind').sum()}/{crashed['at_fault'].sum()}",
            "new collisions in clean rollouts": f"{(clean['outcome'] == 'new collision').sum()}/{clean.height}",
            "median progress lost, clean rollouts (m)": round(float(clean["progress_lost_m"].median() or 0.0), 1),
        }))
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=1, fmt_str_lengths=20):
            print(result.filter(pl.col("eval_collision").is_not_null() | (pl.col("outcome") != "no collision either way"))
                  .with_columns(pl.col("scene").str.slice(7, 8))
                  .select("scene", "at_fault", "eval_collision", "overrides", "logged_collision_s", "logged_actor",
                          "filtered_collision_s", "filtered_actor", "filtered_from_behind", "outcome"))


if __name__ == "__main__":
    main()
