"""Which corridor model finds the obstacles the ego actually drives at (supplementary, path-model table).

On the finished Alpamayo 1.5 rollouts, at every Drive call where the ego turns
(|yaw rate| > YAW_RATE_RAD_S) or changes lane (moves > LANE_CHANGE_M sideways in
the next HORIZON_S), each true obstacle up to AHEAD_M ahead is judged in or out
of the path by three corridor models:

  straight      the ego's current heading
  parabola      a constant-curvature arc from the current yaw rate and speed
  interpolated  the driven plan's waypoints (the validator's model)

against the reference: the ego's actual future path over the next HORIZON_S.
Each counts as in path when its side comes within the validator's corridor
(ego half-width + obstacle half-width + lateral margin) of the path.

Usage (repo root):  uv run python benchmark/path_model_ablation.py
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
for _pool in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
    os.environ.setdefault(_pool, "1")

import numpy as np
import polars as pl
from alpasim_driver.schema import SafetyValidatorConfig
from alpasim_utils.geometry import pose_from_grpc

sys.path.insert(0, str(Path(__file__).parent))
import replay_validator as rv  # noqa: E402

YAW_RATE_RAD_S = 0.05
LANE_CHANGE_M = 1.5
HORIZON_S = 3.0
AHEAD_M = 40.0
SETS = [
    (Path("benchmark/runs_core"), "alpamayo1_5"),
    (Path("benchmark/runs"), "alpamayo1_5"),
    (Path("benchmark/runs"), "alpamayo1_5_vcache_exact"),
    (Path("benchmark/runs"), "alpamayo1_5_vcache_neardup"),
]
CFG = SafetyValidatorConfig(enabled=True)


def _lateral_at(path_xy: np.ndarray, x: float) -> float | None:
    """The (N, 2) rig-frame path's lateral offset at forward distance ``x``; None
    where the path does not reach that far."""
    forward = np.maximum.accumulate(path_xy[:, 0])
    if x > forward[-1]:
        return None
    return float(np.interp(x, forward, path_xy[:, 1]))


def _one(asl_path: Path) -> list[dict]:
    log = asyncio.run(rv._read_log(asl_path))
    traffic = rv._traffic(log)
    ego_times = np.array(sorted(log.ego_poses))
    rows = []
    for drive in log.drives:
        t = drive.selected.poses[0].timestamp_us
        if t not in log.ego_speed:
            continue
        rig_from_local = np.linalg.inv(pose_from_grpc(drive.selected.poses[0].pose).as_se3().astype(np.float64))
        future = ego_times[(ego_times > t) & (ego_times <= t + HORIZON_S * 1e6)]
        if len(future) < 5:
            continue
        actual = np.array([(rig_from_local @ pose_from_grpc(log.ego_poses[u]).as_se3())[:2, 3] for u in future])
        speed, yaw_rate = log.ego_speed[t], log.ego_yaw_rate[t]
        turning = abs(yaw_rate) > YAW_RATE_RAD_S
        lane_change = float(np.max(np.abs(actual[:, 1]))) > LANE_CHANGE_M
        if not (turning or lane_change) or speed < 1.0:
            continue
        prediction = rv._prediction(drive, rig_from_local)
        plan = prediction.candidate_positions[prediction.selected_index, :, :2]
        curvature = yaw_rate / speed
        for obstacle in rv._gt_obstacles(traffic, t, rig_from_local):
            x, y = obstacle["x"], obstacle["y"]
            gap = x - CFG.ego_front_m - obstacle["half_length_m"]
            if gap <= 0.0 or x > AHEAD_M:
                continue
            truth_y = _lateral_at(actual, x)
            if truth_y is None:
                continue  # the ego does not get this far within the horizon
            half = CFG.ego_half_width_m + obstacle["half_width_m"] + CFG.lateral_margin_m
            models = {
                "straight": 0.0,
                "parabola": 0.5 * curvature * x * x,
                "interpolated": _lateral_at(plan, x),
            }
            row = {
                "scene": asl_path.parents[1].name, "time_us": t, "turning": turning,
                "lane_change": lane_change, "truth_in_path": abs(y - truth_y) <= half,
            }
            for name, model_y in models.items():
                row[f"{name}_in_path"] = model_y is not None and abs(y - model_y) <= half
            rows.append(row)
    return rows


def main() -> None:
    paths = [
        p for runs_dir, arm in SETS for d in runs_dir.glob(f"{arm}*")
        if d.name == arm or d.name.startswith(f"{arm}__p")
        for p in d.glob("rollouts/*/*/rollout.asl")
    ]
    with ProcessPoolExecutor(4) as pool:
        rows = [row for rollout_rows in pool.map(_one, paths) for row in rollout_rows]
    df = pl.DataFrame(rows)
    out = Path("benchmark/final/path_model_ablation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    truth_in = df["truth_in_path"]
    print(f"{len(paths)} rollouts, {df.select('scene', 'time_us').unique().height} turning or lane-change steps, "
          f"{df.height} obstacle judgements, {int(truth_in.sum())} truly in path")
    table = []
    for name in ("straight", "parabola", "interpolated"):
        model_in = df[f"{name}_in_path"]
        missed = int((truth_in & ~model_in).sum())
        false = int((~truth_in & model_in).sum())
        table.append({
            "corridor model": name,
            "missed in-path": f"{missed} / {int(truth_in.sum())} ({100 * missed / max(int(truth_in.sum()), 1):.1f}%)",
            "false in-path": f"{false} / {int((~truth_in).sum())} ({100 * false / max(int((~truth_in).sum()), 1):.1f}%)",
        })
    with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True,
                   fmt_str_lengths=40):
        print(pl.DataFrame(table))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
