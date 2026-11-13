"""Summarize benchmark arms into one comparison table.

Reads, per arm directory under RUNS_DIR (default benchmark/runs/):
  driver/latency.jsonl                 per-Drive latency and GPU memory (driver)
  rollouts/<scene>/<rollout>/timing.json   per-rollout wall-clock throughput (runtime)
  aggregate/results-summary.json       per-rollout driving scores (eval)

An arm split into parts (ARM__p1, ARM__p2, ... from benchmark/submit.sh)
counts as one arm once every part has its aggregate results.

Usage (repo root):  uv run python benchmark/summarize.py [--warmup N] [RUNS_DIR]
Writes RUNS_DIR/results.csv (benchmark/results.csv for the default) and prints
a Markdown table.
"""

import argparse
import json
from pathlib import Path

import polars as pl

DEFAULT_RUNS_DIR = Path("benchmark/runs")
BUDGET_MS = 100.0
STAGE_COLUMNS = [
    "input_prep_ms",
    "preprocess_ms",
    "vision_encoder_ms",
    "llm_prefill_ms",
    "cot_decode_ms",
    "trajectory_ms",
]


def latency_summary(run_dirs: list[Path], warmup: int) -> dict[str, float]:
    # Each driver process restarts inference_index at 0, so the first
    # `warmup` inferences (CUDA init, allocator growth) are excluded.
    df = pl.concat(
        [pl.read_ndjson(d / "driver" / "latency.jsonl") for d in run_dirs],
        how="diagonal_relaxed",
    ).filter(pl.col("inference_index") >= warmup)
    inference = df["inference_ms"]
    summary = {
        "drive_calls": float(df.height),
        "mean_batch_size": df["batch_size"].mean(),
        "inference_ms_median": inference.median(),
        "inference_ms_mean": inference.mean(),
        "inference_ms_p95": inference.quantile(0.95),
        "pct_within_100ms": 100.0 * (inference <= BUDGET_MS).mean(),
        "rpc_ms_median": df["rpc_ms"].median(),
        "queue_wait_ms_median": df["queue_wait_ms"].median(),
        "peak_reserved_gib": df["peak_reserved_gib"].max(),
    }
    for column in STAGE_COLUMNS + ["cot_decode_steps"]:
        if column in df.columns:
            summary[f"{column.removesuffix('_ms')}_median"] = df[column].median()
    # Safety validator decisions (arms with driver.safety_validator.enabled).
    if "safety_reason" in df.columns:
        reason = df["safety_reason"]
        summary.update(
            {
                "safety_ms_median": df["safety_ms"].median(),
                # Overlaps model inference; absent in runs predating the overlap.
                "safety_sense_ms_median": (
                    df["safety_sense_ms"].median()
                    if "safety_sense_ms" in df.columns
                    else float("nan")
                ),
                "safety_speed_override_pct": 100.0
                * (reason == "speed_override").mean(),
                "safety_switched_candidate_pct": 100.0
                * df["safety_switched_candidate"].mean(),
                "safety_no_sensor_data_pct": 100.0
                * (reason == "no_sensor_data").mean(),
            }
        )
    # Weather-adaptive arms: how often each safety sensor mode was used.
    if "sensor_mode" in df.columns:
        for mode in ("none", "lidar", "radar", "fused"):
            summary[f"sensor_mode_{mode}_pct"] = (
                100.0 * (df["sensor_mode"] == mode).mean()
            )
    # Visual feature cache counters (images requested / encoded / reused).
    for column in df.columns:
        if column.startswith("vision_") and column != "vision_encoder_ms":
            summary[f"{column}_mean"] = df[column].mean()
    return summary


def throughput_summary(run_dirs: list[Path]) -> dict[str, float]:
    timings = [
        json.loads(path.read_text())
        for d in run_dirs
        for path in d.glob("rollouts/*/*/timing.json")
    ]
    df = pl.DataFrame(timings)
    return {
        "sim_steps_per_s": df["sim_steps_per_s"].mean(),
        "realtime_ratio": df["realtime_ratio"].mean(),
    }


def driving_summary(run_dirs: list[Path]) -> dict[str, float]:
    rollouts = [
        rollout
        for d in run_dirs
        for rollout in json.loads(
            (d / "aggregate" / "results-summary.json").read_text()
        )["rollouts"]
    ]
    df = pl.DataFrame(
        [
            {
                "score": rollout["score"],
                "passed": rollout["passed"],
                "collision_at_fault": rollout["metrics"]["collision_at_fault"],
                "collision_any": rollout["metrics"]["collision_any"],
                "offroad": rollout["metrics"]["offroad"],
                "progress_rel": rollout["metrics"]["progress_rel"],
                "progress_clipped_rel": rollout["metrics"]["progress_clipped_rel"],
            }
            for rollout in rollouts
        ]
    )
    return {
        "rollouts": float(df.height),
        "scene_score_mean": df["score"].mean(),
        "pass_rate_pct": 100.0 * df["passed"].mean(),
        # Progress only counts while the ego stays near the GT route; this is
        # the progress that enters scene_score.
        "route_completion_pct": 100.0 * df["progress_clipped_rel"].mean(),
        "distance_completion_pct": 100.0 * df["progress_rel"].mean(),
        "collision_at_fault_rate_pct": 100.0 * (df["collision_at_fault"] > 0).mean(),
        "collision_any_rate_pct": 100.0 * (df["collision_any"] > 0).mean(),
        "offroad_rate_pct": 100.0 * (df["offroad"] > 0).mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--warmup", type=int, default=3, help="inferences to drop per driver process"
    )
    parser.add_argument("runs_dir", nargs="?", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()
    runs_dir = args.runs_dir

    arms: dict[str, list[Path]] = {}
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if (run_dir / "generated-user-config-0.yaml").exists():
            arms.setdefault(run_dir.name.split("__p")[0], []).append(run_dir)

    rows = []
    for arm, run_dirs in arms.items():
        unfinished = [
            d.name
            for d in run_dirs
            if not (d / "aggregate" / "results-summary.json").exists()
        ]
        if unfinished:
            print(f"Skipping {arm}: no aggregate results yet in {', '.join(unfinished)}")
            continue
        rows.append(
            {
                "arm": arm,
                "parts": float(len(run_dirs)),
                **driving_summary(run_dirs),
                **latency_summary(run_dirs, args.warmup),
                **throughput_summary(run_dirs),
            }
        )
    if not rows:
        raise SystemExit(f"No finished arms under {runs_dir}")

    table = pl.DataFrame(rows)
    output = (
        Path("benchmark/results.csv")
        if runs_dir == DEFAULT_RUNS_DIR
        else runs_dir / "results.csv"
    )
    table.write_csv(output)
    print(f"Wrote {output}")
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        float_precision=2,
        tbl_cols=-1,
        tbl_rows=-1,
    ):
        # Transpose so arms are columns: easier to read with many metrics.
        print(table.transpose(include_header=True, column_names="arm"))


if __name__ == "__main__":
    main()
