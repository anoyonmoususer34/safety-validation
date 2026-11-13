"""Why the validator's rear check did not flag the rear-ends in the finished
Alpamayo 1.5 rollouts: where the following car was, how fast it
closed, and which checks fired in the REAR_WINDOW_S before the collision.

Usage (repo root):  uv run python benchmark/rear_collisions.py REPLAY_NAME
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from alpasim_driver.schema import SafetyValidatorConfig
from alpasim_utils.geometry import pose_from_grpc

sys.path.insert(0, str(Path(__file__).parent))
import counterfactual_braking as cf  # noqa: E402
import replay_validator as rv  # noqa: E402

REAR_WINDOW_S = 3.0
SETS = [(Path("benchmark/runs_core"), "alpamayo1_5", ""), (Path("benchmark/runs"), "alpamayo1_5", "_alpamayo1_5"),
        (Path("benchmark/runs"), "alpamayo1_5_vcache_exact", "_alpamayo1_5_vcache_exact"),
        (Path("benchmark/runs"), "alpamayo1_5_vcache_neardup", "_alpamayo1_5_vcache_neardup")]
CFG = SafetyValidatorConfig(enabled=True)


def main() -> None:
    replay_name = sys.argv[1]
    rows = []
    for runs_dir, arm, suffix in SETS:
        steps = pl.read_csv(runs_dir / f"{replay_name}{suffix}" / "steps_fused.csv", infer_schema_length=None)
        asl = {p.parent.name: p for p in runs_dir.glob(f"{arm}*/rollouts/*/*/rollout.asl")
               if p.parents[3].name == arm or p.parents[3].name.startswith(f"{arm}__p")}
        for rollout, path in asl.items():
            log = asyncio.run(rv._read_log(path))
            track, _ = cf.counterfactual_track(log, {})
            hit = cf.first_collision(log, None, track, {})
            if hit is None or not hit[2]:
                continue
            t_hit, actor, _ = hit
            t0 = min(log.actor_poses[cf.EGO_ID])
            size = log.aabbs[actor]
            corridor = CFG.ego_half_width_m + size.y / 2 + CFG.lateral_margin_m
            actor_path = cf.LoggedPath(log.actor_poses[actor])
            row = {"set": arm + suffix.replace(f"_{arm}", ""), "scene": path.parents[1].name[7:15],
                   "collision_s": round((t_hit - t0) * 1e-6, 1), "actor": log.labels[actor]}
            for before_s in (2.0, 1.0):
                t = t_hit - int(before_s * 1e6)
                tt, xy, yaw, ego_v, _ = min(track, key=lambda s: abs(s[0] - t))
                axy = actor_path.at(actor_path.s_at(tt))[0]
                c, si = math.cos(yaw), math.sin(yaw)
                x = c * (axy[0] - xy[0]) + si * (axy[1] - xy[1])
                y = -si * (axy[0] - xy[0]) + c * (axy[1] - xy[1])
                gap = -x - CFG.ego_rear_m - size.x / 2
                closing = actor_path.speed_at(tt) - ego_v
                row[f"{before_s:g}s_before"] = (
                    f"x={x:+.1f} y={y:+.1f} gap={gap:.1f} m, closing {closing:.1f} m/s, "
                    f"{'in' if abs(y) <= corridor and -x <= CFG.rear_prediction_horizon else 'OUT of'} rear corridor"
                )
                row[f"ego_mps_{before_s:g}s"] = round(ego_v, 1)
            window = steps.filter((pl.col("rollout") == rollout) & (pl.col("time_us") >= t_hit - REAR_WINDOW_S * 1e6)
                                  & (pl.col("time_us") <= t_hit))
            flags = window["model_violations"].fill_null("").str.split("|").explode().unique().to_list()
            row["checks fired in 3 s before (model plan)"] = ", ".join(sorted(f for f in flags if f)) or "none"
            rows.append(row)
    df = pl.DataFrame(rows).sort("set", "scene")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=260, fmt_str_lengths=90,
                   tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(df)
    df.write_csv(Path("benchmark/final") / "rear_collisions.csv")


if __name__ == "__main__":
    main()
