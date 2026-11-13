"""Tables for the safety validator against its baselines, and its sensitivity sweep.

Prevented collisions come from benchmark/hazard_stress.py (fused LiDAR + radar,
following cars that react after 1.0 s); unnecessary interventions from
benchmark/replay_validator.py on the finished Alpamayo 1.5 rollouts that did not
crash.  The validator's default row is closed_loop_followers.csv and
validator_replay_v13; the others come from benchmark/run_baselines_and_sweep.sh.

Usage (repo root):  uv run python benchmark/baseline_sweep_tables.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

HAZARDS = Path("benchmark/hazard_stress")
# Finished Alpamayo 1.5 rollouts: runs directory and replay-name suffix.
REPLAY_SETS = [
    (Path("benchmark/runs_core"), ""),
    (Path("benchmark/runs"), "_alpamayo1_5"),
    (Path("benchmark/runs"), "_alpamayo1_5_vcache_exact"),
    (Path("benchmark/runs"), "_alpamayo1_5_vcache_neardup"),
]
FAMILIES = {"lead_brake": "braking lead", "cut_in": "cut-in", "ped": "pedestrian"}
ROWS = {
    # tag: (hazard-test file, replay name, label)
    "validator": ("closed_loop_followers.csv", "validator_replay_v13", "Ours (fused)"),
    "baseline_ttc": ("closed_loop_baseline_ttc.csv", "validator_replay_baseline_ttc", "TTC-only braking"),
    "baseline_rss": ("closed_loop_baseline_rss.csv", "validator_replay_baseline_rss", "RSS"),
}
SWEEP = {
    "validator": "defaults (beta 2, horizon 3 s, max decel 6)",
    "sweep_beta0": "beta 0",
    "sweep_beta1": "beta 1",
    "sweep_horizon1": "crossing horizon 1 s",
    "sweep_horizon2": "crossing horizon 2 s",
    "sweep_horizon4": "crossing horizon 4 s",
    "sweep_decel4": "max decel 4 m/s^2",
    "sweep_decel8": "max decel 8 m/s^2",
}


def _pct(n: int, d: int) -> str:
    return f"{n} / {d} ({100 * n / d:.0f}%)" if d else "-"


def hazard_metrics(path: Path, model: str | None) -> dict:
    """Prevented collisions with the hazard, overall and per family, and new
    collisions with other road users when following cars react after 1.0 s."""
    r = pl.read_csv(path, infer_schema_length=None).filter(
        pl.col("applicable") & (pl.col("sensor") == "fused")
    )
    if model:
        r = r.filter(pl.col("model") == model)
    r = r.with_columns(
        pl.col("hazard").str.replace(r"^(lead_brake|cut_in|ped)_.*$", "$1").alias("family"),
        (pl.col("baseline_hits_hazard") & ~pl.col("filtered_hits_hazard")).alias("prevented"),
    )
    out = {
        "prevented": _pct(int(r["prevented"].sum()), int(r["baseline_hits_hazard"].sum())),
        "new collisions (followers react 1.0 s)": int(r["new_other_collision_react_1s"].sum()),
    }
    for family, label in FAMILIES.items():
        f = r.filter(pl.col("family") == family)
        out[label] = _pct(int(f["prevented"].sum()), int(f["baseline_hits_hazard"].sum()))
    return out


def intervention_metrics(replay_name: str) -> dict:
    """Interventions in the finished Alpamayo rollouts that did not crash."""
    frames = []
    for runs_dir, suffix in REPLAY_SETS:
        path = runs_dir / f"{replay_name}{suffix}" / "rollouts_fused.csv"
        if not path.exists():
            return {"steps acted on": "missing", "rollouts with an emergency stop": "missing"}
        frames.append(pl.read_csv(path, infer_schema_length=None))
    r = pl.concat(frames, how="vertical_relaxed").filter(pl.col("collision").is_null())
    acted = (r["speed_override_pct"] * r["steps"]).sum() / r["steps"].sum()
    return {
        "steps acted on": f"{acted:.1f}%",
        "rollouts with an emergency stop": _pct(int((r["emergency_ttc_pct"] > 0).sum()), r.height),
    }


def _table(rows: list[dict]) -> None:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=60,
                   tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        print(pl.DataFrame(rows))


def main() -> None:
    print("## Validator vs. baselines (fused LiDAR + radar)\n")
    rows = []
    for tag, (hazard_file, replay_name, label) in ROWS.items():
        path = HAZARDS / hazard_file
        if not path.exists():
            continue
        for model in (None, "Alpamayo 1.5"):
            rows.append({
                "method": label, "policies": model or "all three",
                **hazard_metrics(path, model), **intervention_metrics(replay_name),
            })
    _table(rows)

    print("\n## Sensitivity sweep (Alpamayo 1.5, fused)\n")
    rows = []
    for tag, label in SWEEP.items():
        path = HAZARDS / (ROWS["validator"][0] if tag == "validator" else f"closed_loop_{tag}.csv")
        replay = ROWS["validator"][1] if tag == "validator" else f"validator_replay_{tag}"
        if path.exists():
            rows.append({"setting": label, **hazard_metrics(path, "Alpamayo 1.5"), **intervention_metrics(replay)})
    _table(rows)


if __name__ == "__main__":
    main()
