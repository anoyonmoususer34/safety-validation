"""What the validator intervenes on in normal Alpamayo 1.5 driving (supplementary, normal-driving and reasoning tables).

Replays the validator with fused sensing over the finished Alpamayo 1.5 rollouts
(as benchmark/replay_validator.py does) and, at every intervention (speed
override), finds the binding obstacle -- the flagged obstacle with the lowest safe
speed -- and matches it to the nearest true road user.  Writes one row per
intervention with the policy's logged reasoning text, then prints:

  false positives by cause   interventions in collision-free driving (before any
                             collision), by the binding obstacle's source and match
  reasoning vs. intervention whether the reasoning names the hazard class of the
                             binding obstacle and says to slow or stop (keyword match)

and writes a random sample of 50 interventions for hand-labelling the keyword match.

Usage (repo root):  uv run python benchmark/normal_driving_interventions.py
"""

from __future__ import annotations

import asyncio
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _pool in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_pool, "1")

import numpy as np
import polars as pl
from alpasim_driver.safety_filter import SafetyFilter
from alpasim_utils.geometry import pose_from_grpc
from alpasim_utils.logs import async_read_pb_log

sys.path.insert(0, str(Path(__file__).parent))
import replay_validator as rv  # noqa: E402

SETS = [(Path("benchmark/runs_core"), "alpamayo1_5"), (Path("benchmark/runs"), "alpamayo1_5"),
        (Path("benchmark/runs"), "alpamayo1_5_vcache_exact"), (Path("benchmark/runs"), "alpamayo1_5_vcache_neardup")]
OUT = Path("benchmark/final")
MATCH_M = 3.0  # binding obstacle centre to a true road user's centre
STATIONARY_MPS = 0.5
IN_LANE_M = 1.5  # |path lateral error| of a lead vehicle
VEHICLE = re.compile(r"\b(car|cars|vehicle|vehicles|truck|bus|van|lead|traffic)\b", re.I)
CUT_IN = re.compile(r"\b(merg\w*|cut\w*|chang\w* lanes?|lane change)\b", re.I)
PEDESTRIAN = re.compile(r"\b(pedestrian\w*|person|people|walker|crossing)\b", re.I)
SLOW = re.compile(r"\b(slow\w*|stop\w*|decelerat\w*|brak\w*|yield\w*|wait\w*)\b", re.I)


async def _reasoning(asl_path: Path) -> dict[int, str]:
    """The policy's reasoning text per Drive request time."""
    out, request_us = {}, None
    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_request":
            request_us = entry.driver_request.time_now_us
        elif kind == "driver_return" and request_us is not None:
            debug = pickle.loads(entry.driver_return.debug_info.unstructured_debug_info)
            out[request_us] = debug.get("reasoning_text") or ""
    return out


def _binding(debug: dict) -> dict | None:
    """The flagged obstacle with the lowest safe speed."""
    trace = debug["validator_result"]["trace_data"]
    flagged = [a for a in trace.get("per_obstacle_assessment", []) if a["flags"]]
    if not flagged:
        return None
    binding = min(flagged, key=lambda a: (a["safe_speed"], a["confidence_adjusted_ttc"]))
    where = {o["id"]: o for o in trace["relevant_obstacles"]}[binding["id"]]
    return {**binding, "crossing": where["crossing"]}


def _one(job: dict) -> list[dict]:
    asl_path = job["asl_path"]
    log = asyncio.run(rv._read_log(asl_path))
    reasoning = asyncio.run(_reasoning(asl_path))
    request_times = np.array(sorted(reasoning))
    collision_us, _ = rv._first_collision(asl_path.parent / "metrics.parquet")
    traffic = rv._traffic(log)
    ego_trajectory = rv._trajectory(log.ego_poses)
    safety_filter = SafetyFilter(rv.validator_config([]), rv.OUTPUT_FREQUENCY_HZ)
    rows, steps = [], 0
    for drive in log.drives:
        time_us = drive.selected.poses[0].timestamp_us
        if time_us not in log.ego_speed:
            continue
        speed = log.ego_speed[time_us]
        rig_from_local = np.linalg.inv(pose_from_grpc(drive.selected.poses[0].pose).as_se3().astype(np.float64))
        sensed = rv.sense(safety_filter, "fused", traffic, ego_trajectory, time_us, rig_from_local, speed,
                          log.ego_yaw_rate[time_us], job["radars"], 0, mode=rv.FIXED_MODES["fused"])
        _, debug = safety_filter.apply(rv._prediction(drive, rig_from_local), sensed, speed)
        before_collision = collision_us is None or time_us < collision_us
        steps += before_collision
        if debug["reason"] != "speed_override":
            continue
        binding = _binding(debug)
        row = {"scene": asl_path.parents[1].name, "rollout": asl_path.parent.name, "time_us": time_us,
               "at_fault_rollout": bool(job["at_fault"]), "before_collision": bool(before_collision), "speed_mps": speed,
               "safe_speed_mps": debug["safe_speed"], "violations": "|".join(debug["violations"])}
        if binding is not None:
            # Its sensed centre (the trace's distances are bumper gaps).
            centre = next(o for o in sensed.obstacles if o["id"] == binding["id"])
            binding |= {"x": centre["x"], "y": centre["y"]}
            truth = rv._gt_obstacles(traffic, time_us, rig_from_local)
            nearest = min(truth, key=lambda o: np.hypot(o["x"] - binding["x"], o["y"] - binding["y"]), default=None)
            matched = bool(nearest is not None and np.hypot(nearest["x"] - binding["x"], nearest["y"] - binding["y"]) < MATCH_M)
            row |= {"source": binding["source"], "confidence": binding["confidence"], "x": binding["x"],
                    "y": binding["y"], "path_lateral_error": binding["path_lateral_error"], "crossing": bool(binding["crossing"]),
                    "matched": matched,
                    "truth_label": log.labels.get(nearest["id"], "") if matched else None,
                    "truth_speed_mps": float(np.hypot(nearest["vx"], nearest["vy"])) if matched else None}
        k = int(np.searchsorted(request_times, time_us, side="right")) - 1
        row["reasoning"] = reasoning[int(request_times[k])] if k >= 0 else ""
        rows.append(row)
    return [{"__steps__": steps, "at_fault_rollout": job["at_fault"]}] + rows


def cause(df: pl.DataFrame) -> pl.Expr:
    vehicle = pl.col("truth_label").is_in(["automobile", "truck", "bus", "heavy_truck", "trailer"])
    return (
        pl.when(pl.col("source").is_null()).then(pl.lit("Other: no flagged obstacle"))
        .when(pl.col("source") == "radar").then(pl.lit("Radar clutter or radar-only obstacle"))
        .when(~pl.col("matched")).then(pl.lit("LiDAR obstacle with no road user (ghost track)"))
        .when(vehicle & (pl.col("truth_speed_mps") < STATIONARY_MPS) & (pl.col("path_lateral_error").abs() > IN_LANE_M))
        .then(pl.lit("Parked vehicle near the corridor"))
        .when(~vehicle & (pl.col("truth_label") != "person")).then(pl.lit("Roadside or static object"))
        .otherwise(pl.lit("Road user in the path (lead, merging car, pedestrian)"))
    )


def hazard_class() -> pl.Expr:
    vehicle = pl.col("truth_label").is_in(["automobile", "truck", "bus", "heavy_truck", "trailer"])
    return (
        pl.when(pl.col("truth_label") == "person").then(pl.lit("Pedestrian"))
        .when(vehicle & (pl.col("crossing") | (pl.col("path_lateral_error").abs() > IN_LANE_M))).then(pl.lit("Cut-in"))
        .when(vehicle).then(pl.lit("Lead vehicle"))
        .otherwise(None)
    )


def md(df: pl.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()])


def main() -> None:
    shared = {"radars": rv._load_radars(rv.DEFAULT_RADAR_CONFIG)}
    jobs = [j for runs_dir, arm in SETS for j in rv._jobs(runs_dir, arm, "fused", shared)]
    print(f"{len(jobs)} rollouts")
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(_one, jobs))
    clean_steps = sum(r[0]["__steps__"] for r in results if not r[0]["at_fault_rollout"])
    df = pl.DataFrame([row for r in results for row in r[1:]], infer_schema_length=None)
    df = df.with_columns(cause=cause(df), hazard=hazard_class())
    df.write_csv(OUT / "normal_driving_interventions.csv")

    minutes = clean_steps / rv.OUTPUT_FREQUENCY_HZ / 60
    fp = df.filter(~pl.col("at_fault_rollout") & pl.col("before_collision"))
    print(f"\n## False positives by cause: {fp.height} interventions in {minutes:.1f} min of collision-free driving "
          f"({len([r for r in results if not r[0]['at_fault_rollout']])} rollouts)\n")
    t = fp.group_by("cause").len().sort("len", descending=True).with_columns(per_minute=(pl.col("len") / minutes).round(2))
    print(md(t))

    named = (
        pl.when(pl.col("hazard") == "Pedestrian").then(pl.col("reasoning").str.contains(PEDESTRIAN.pattern))
        .when(pl.col("hazard") == "Cut-in").then(pl.col("reasoning").str.contains(CUT_IN.pattern)
                                                 | pl.col("reasoning").str.contains(VEHICLE.pattern))
        .otherwise(pl.col("reasoning").str.contains(VEHICLE.pattern))
    )
    r = df.filter(pl.col("hazard").is_not_null()).with_columns(
        named=named, slow=pl.col("reasoning").str.contains(SLOW.pattern))
    rows = r.group_by("hazard").agg(n=pl.len(), named=(100 * pl.col("named").mean()).round(0),
                                    slow=(100 * pl.col("slow").mean()).round(0)).sort("hazard")
    total = r.select(hazard=pl.lit("Total"), n=pl.len(), named=(100 * pl.col("named").mean()).round(0),
                     slow=(100 * pl.col("slow").mean()).round(0))
    print(f"\n## Reasoning vs. intervention ({r.height} interventions with a matched road user, all rollouts)\n")
    print(md(pl.concat([rows, total.cast(rows.schema)])))
    r.sample(min(50, r.height), seed=0).select(
        "scene", "time_us", "hazard", "reasoning", "named", "slow",
        hand_named=pl.lit(None, pl.Boolean), hand_slow=pl.lit(None, pl.Boolean),
    ).write_csv(OUT / "reasoning_handlabel_sample.csv")
    print(f"\nWrote {OUT / 'normal_driving_interventions.csv'} and {OUT / 'reasoning_handlabel_sample.csv'}")


if __name__ == "__main__":
    main()
