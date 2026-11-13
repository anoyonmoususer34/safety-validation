"""The baseline, sweep, selection, constraint, comfort and miss tables of the
supplementary, from the outputs of benchmark/reproduce_cpu.sh.  Prints Markdown
and writes benchmark/final/tables.md.

  python benchmark/final_tables.py            all tables whose inputs exist
  python benchmark/final_tables.py t1 mcnemar only these

Hazard test: fused LiDAR + radar unless a table says otherwise; following cars
react after 1.0 s.  Normal driving: the finished Alpamayo 1.5 rollouts that did
not crash, replayed.
"""

from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
from alpasim_driver.safety_filter import override_decel
from alpasim_driver.schema import SafetyValidatorConfig

FINAL = Path("benchmark/final")
FAMILIES = {"lead_brake": "braking lead", "cut_in": "cut-in", "ped": "pedestrian"}
SENSORS = ("gt", "radar", "lidar", "fused")
MODELS = ("Alpamayo 1.5", "SimForge-D1.5", "VaVAM")
ALPAMAYO_REPLAYS = [  # runs directory, replay-name suffix
    (Path("benchmark/runs_core"), ""),
    (Path("benchmark/runs"), "_alpamayo1_5"),
    (Path("benchmark/runs"), "_alpamayo1_5_vcache_exact"),
    (Path("benchmark/runs"), "_alpamayo1_5_vcache_neardup"),
]
BOOTSTRAP = 10_000
CFG = SafetyValidatorConfig(enabled=True)


def _print(df: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=260, fmt_str_lengths=80, tbl_formatting="MARKDOWN",
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(df)


def _pct(n: int, d: int) -> str:
    return f"{n} / {d} ({100 * n / d:.1f}%)" if d else "-"


def hazard(name: str) -> pl.DataFrame:
    return pl.read_csv(FINAL / f"hazard_{name}.csv", infer_schema_length=None).filter(pl.col("applicable")).with_columns(
        pl.col("hazard").str.replace(r"^(lead_brake|cut_in|ped)_.*$", "$1").alias("family"),
        (pl.col("baseline_hits_hazard") & ~pl.col("filtered_hits_hazard")).alias("prevented"),
    )


def _scene_ci(df: pl.DataFrame, rng: np.random.Generator) -> str:
    """95% interval of the prevention rate, resampling whole scenes."""
    per_scene = df.filter(pl.col("baseline_hits_hazard")).group_by("scene").agg(
        pl.col("prevented").sum().alias("p"), pl.len().alias("n"))
    p, n = per_scene["p"].to_numpy(), per_scene["n"].to_numpy()
    draws = rng.integers(0, len(p), size=(BOOTSTRAP, len(p)))
    rates = p[draws].sum(axis=1) / n[draws].sum(axis=1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return f"{100 * lo:.1f}-{100 * hi:.1f}%"


def _new_collisions(df: pl.DataFrame, tag: str) -> str:
    new = df[f"new_other_collision{tag}"]
    behind = df[f"new_other_from_behind{tag}"].fill_null(False)
    return f"{int(new.sum())} ({int((new & behind).sum())} from behind)"


def table1() -> None:
    """Table 1: prevented collisions by policy and sensors."""
    df = hazard("main")
    rng = np.random.default_rng(0)
    rows = []
    for model in (None, *MODELS):
        d = df if model is None else df.filter(pl.col("model") == model)
        for sensor in SENSORS:
            s = d.filter(pl.col("sensor") == sensor)
            rows.append({
                "policy": model or "all three", "sensors": sensor,
                "prevented": _pct(int(s["prevented"].sum()), int(s["baseline_hits_hazard"].sum())),
                "95% CI (scenes resampled)": _scene_ci(s, rng),
                "new collisions, followers react 1.0 s": _new_collisions(s, "_react_1s"),
                "new collisions, followers react 1.5 s": _new_collisions(s, "_react_1.5s"),
                "new collisions, recorded followers": _new_collisions(s, ""),
            })
    print("## Table 1: injected hazards prevented (closed-loop validator, logged policy plans)\n")
    _print(pl.DataFrame(rows))


def crossing() -> None:
    """Crossing-prediction ablation, same cases."""
    keys = ["scene", "rollout", "hazard", "sensor"]
    on, off = hazard("main"), hazard("no_crossing")
    j = on.select(*keys, "model", "family", "baseline_hits_hazard", pl.col("prevented").alias("with")).join(
        off.select(*keys, pl.col("prevented").alias("without")), on=keys)
    rows = []
    for model in (None, "Alpamayo 1.5"):
        d = j if model is None else j.filter(pl.col("model") == model)
        for sensor in SENSORS:
            for family, label in FAMILIES.items():
                s = d.filter((pl.col("sensor") == sensor) & (pl.col("family") == family))
                base = int(s["baseline_hits_hazard"].sum())
                rows.append({"policy": model or "all three", "sensors": sensor, "hazard": label,
                             "without crossing": _pct(int(s["without"].sum()), base),
                             "with crossing": _pct(int(s["with"].sum()), base)})
    print("## Crossing-prediction ablation\n")
    _print(pl.DataFrame(rows))


def per_hazard() -> None:
    """Per-hazard-variant table."""
    df = hazard("main").filter(pl.col("sensor") == "fused")
    rows = []
    for model in ("Alpamayo 1.5", None):
        d = df if model is None else df.filter(pl.col("model") == model)
        for hazard_name in sorted(d["hazard"].unique().to_list()):
            s = d.filter(pl.col("hazard") == hazard_name)
            rows.append({"policy": model or "all three", "hazard": hazard_name, "cases": s.height,
                         "prevented": _pct(int(s["prevented"].sum()), int(s["baseline_hits_hazard"].sum()))})
    print("## Per hazard variant (fused)\n")
    _print(pl.DataFrame(rows))


def _normal_driving(replay: str) -> dict:
    frames = [pl.read_csv(d / f"{replay}{suffix}" / "rollouts_fused.csv", infer_schema_length=None)
              for d, suffix in ALPAMAYO_REPLAYS]
    r = pl.concat(frames, how="vertical_relaxed").filter(pl.col("collision").is_null())
    acted = (r["speed_override_pct"] * r["steps"]).sum() / r["steps"].sum()
    return {"steps acted on (normal driving)": f"{acted:.1f}%",
            "rollouts with an emergency stop": _pct(int((r["emergency_ttc_pct"] > 0).sum()), r.height)}


def baselines() -> None:
    """Baseline table: ours vs. TTC-only braking and RSS."""
    rows = []
    for label, name, replay in (("Ours", "main", "validator_replay_final"),
                                ("TTC-only braking", "baseline_ttc", "validator_replay_final_baseline_ttc"),
                                ("RSS", "baseline_rss", "validator_replay_final_baseline_rss")):
        df = hazard(name).filter(pl.col("sensor") == "fused")
        normal = _normal_driving(replay)
        for model in (None, "Alpamayo 1.5"):
            d = df if model is None else df.filter(pl.col("model") == model)
            row = {"method": label, "policy": model or "all three",
                   "prevented": _pct(int(d["prevented"].sum()), int(d["baseline_hits_hazard"].sum()))}
            for family, f_label in FAMILIES.items():
                s = d.filter(pl.col("family") == family)
                row[f_label] = _pct(int(s["prevented"].sum()), int(s["baseline_hits_hazard"].sum()))
            row["new collisions (followers 1.0 s)"] = _new_collisions(d, "_react_1s")
            rows.append({**row, **normal})
    print("## Baselines (fused)\n")
    _print(pl.DataFrame(rows))


def sweep() -> None:
    """Sensitivity sweep (Alpamayo 1.5, fused)."""
    settings = {"main": "defaults (beta 2, horizon 3 s, max decel 6)", "sweep_beta0": "beta 0",
                "sweep_beta1": "beta 1", "sweep_horizon1": "crossing horizon 1 s",
                "sweep_horizon2": "crossing horizon 2 s", "sweep_horizon4": "crossing horizon 4 s",
                "sweep_decel4": "max decel 4 m/s^2", "sweep_decel8": "max decel 8 m/s^2"}
    rows = []
    for name, label in settings.items():
        df = hazard(name).filter((pl.col("sensor") == "fused") & (pl.col("model") == "Alpamayo 1.5"))
        replay = "validator_replay_final" if name == "main" else f"validator_replay_final_{name}"
        row = {"setting": label, "prevented": _pct(int(df["prevented"].sum()), int(df["baseline_hits_hazard"].sum()))}
        for family, f_label in FAMILIES.items():
            s = df.filter(pl.col("family") == family)
            row[f_label] = _pct(int(s["prevented"].sum()), int(s["baseline_hits_hazard"].sum()))
        row["new collisions (followers 1.0 s)"] = _new_collisions(df, "_react_1s")
        rows.append({**row, **_normal_driving(replay)})
    print("## Sensitivity sweep (Alpamayo 1.5, fused)\n")
    _print(pl.DataFrame(rows))


def _steps(replay: str, sensor: str = "fused") -> pl.DataFrame:
    return pl.concat([pl.read_csv(d / f"{replay}{suffix}" / f"steps_{sensor}.csv", infer_schema_length=None)
                      for d, suffix in ALPAMAYO_REPLAYS], how="vertical_relaxed").with_columns(
        pl.col("safe_speed_mps").cast(pl.Float64, strict=False),
        pl.col("model_violations").fill_null(""), pl.col("violations").fill_null(""))


def tiers() -> None:
    """Selection-tier usage: how flagged model plans were resolved."""
    s = _steps("validator_replay_final").filter(pl.col("model_violations") != "")
    resolved = s.with_columns(
        pl.when(pl.col("reason") == "unconfirmed").then(pl.lit("1: model plan kept (awaiting confirmation)"))
        .when(pl.col("reason") == "safe").then(pl.lit("2: switched to a safe candidate"))
        .when(pl.col("switched")).then(pl.lit("3: slowed, on another candidate"))
        .otherwise(pl.lit("3: slowed, on the model's candidate")).alias("tier"))
    t = resolved.group_by("tier").len().sort("tier").with_columns(
        (100 * pl.col("len") / pl.col("len").sum()).round(1).alias("% of flagged steps"))
    print(f"## Selection tiers: {s.height} steps where the model's own plan failed a check "
          "(Alpamayo 1.5, normal driving and crashes, fused)\n")
    _print(t.rename({"len": "steps"}))


def n1_vs_n6() -> None:
    """the validator with one candidate vs. Alpamayo's six (fused, real rollouts)."""
    rows = []
    for label, replay in (("N = 6 (ours)", "validator_replay_final"), ("N = 1", "validator_replay_final_n1")):
        s = _steps(replay)
        override = s.filter(pl.col("reason") == "speed_override")
        reduction = (override["speed_mps"] - override["safe_speed_mps"]).clip(lower_bound=0.0)
        cf = pl.concat([pl.read_csv(d / f"{replay}{suffix}" / "counterfactual_fused.csv", infer_schema_length=None)
                        for d, suffix in ALPAMAYO_REPLAYS], how="vertical_relaxed")
        clean = cf.filter(pl.col("logged_collision_s_react_1s").is_null())
        af = cf.filter(pl.col("at_fault"))
        rows.append({
            "candidates": label,
            "interventions (slowdown steps)": override.height,
            "speed reduction per intervention, median (m/s)": round(float(reduction.median()), 2),
            "route lost per rollout without a crash, mean (m)": round(float(clean["progress_lost_m"].cast(pl.Float64).mean()), 2),
            "at-fault crashes avoided (followers 1.0 s)": _pct(
                int(af["outcome_react_1s"].is_in(["avoided", "only hit from behind"]).sum()), af.height),
            "new crashes (followers 1.0 s)": int((cf["outcome_react_1s"] == "new collision").sum()),
        })
    print("## N = 1 vs N = 6 candidates, Alpamayo 1.5 rollouts, fused\n")
    _print(pl.DataFrame(rows))


def mcnemar() -> None:
    """paired exact McNemar test, ours vs. RSS, on cases the baseline policy hits."""
    keys = ["scene", "rollout", "hazard"]
    ours = hazard("main").filter((pl.col("sensor") == "fused") & pl.col("baseline_hits_hazard"))
    rss = hazard("baseline_rss").filter(pl.col("baseline_hits_hazard"))
    j = ours.select(*keys, pl.col("prevented").alias("ours")).join(rss.select(*keys, pl.col("prevented").alias("rss")), on=keys)
    b = int((j["ours"] & ~j["rss"]).sum())
    c = int((~j["ours"] & j["rss"]).sum())
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n) if n else 1.0
    print(f"## McNemar, ours vs. RSS\n\n{j.height} paired cases; ours only: {b}, RSS only: {c}, "
          f"both: {int((j['ours'] & j['rss']).sum())}, neither: {int((~j['ours'] & ~j['rss']).sum())}; "
          f"exact two-sided p = {p:.2g}\n")


def _decels(df: pl.DataFrame) -> np.ndarray:
    values = df["commanded_decels"].fill_null("").str.split("|").explode()
    return values.filter(values != "").cast(pl.Float64).to_numpy()


def first_constraint() -> None:
    """the first constraint to fire after the hazard's onset, prevented cases."""
    df = hazard("main_fused_detail").filter(pl.col("prevented"))
    df = df.with_columns(pl.col("first_reaction_constraints").fill_null("none").str.split("|").list.first()
                         .str.replace("_VIOLATION", "").alias("first"))
    t = df.group_by("family", "first").len().sort("family", "len", descending=[False, True])
    print("## First constraint to fire, prevented cases, fused, all policies\n")
    _print(t.rename({"len": "cases"}))


def decelerations() -> None:
    """commanded decelerations, prevented and missed hazards and normal driving."""
    df = hazard("main_fused_detail")
    s = _steps("validator_replay_final").filter(pl.col("reason") == "speed_override")
    normal = np.array([override_decel(CFG, v, safe, "EMERGENCY_TTC" in viol)
                       for v, safe, viol in s.select("speed_mps", "safe_speed_mps", "violations").iter_rows()])
    rows = []
    for label, values in (("prevented hazards", _decels(df.filter(pl.col("prevented")))),
                          ("missed hazards", _decels(df.filter(pl.col("baseline_hits_hazard") & ~pl.col("prevented")))),
                          ("normal driving (Alpamayo replays)", normal)):
        rows.append({"cases": label, "slowdown steps": len(values),
                     "median (m/s^2)": round(float(np.median(values)), 2),
                     "95th pct (m/s^2)": round(float(np.percentile(values, 95)), 2),
                     "> 4 m/s^2": f"{100 * float(np.mean(values > 4.0)):.1f}%"})
    print("## Commanded deceleration, fused\n")
    _print(pl.DataFrame(rows))


def impact_speeds() -> None:
    """relative impact speed of the hazards the validator did not prevent."""
    df = hazard("main_fused_detail").filter(pl.col("baseline_hits_hazard") & pl.col("filtered_hits_hazard"))
    base, val = df["baseline_impact_speed_mps"].cast(pl.Float64), df["filtered_impact_speed_mps"].cast(pl.Float64)
    rows = [{"hazard": "all", "misses": df.height}]
    for family, label in FAMILIES.items():
        rows.append({"hazard": label, "misses": int((df["family"] == family).sum())})
    out = []
    for row in rows:
        mask = np.ones(df.height, bool) if row["hazard"] == "all" else (df["family"] == [k for k, v in FAMILIES.items() if v == row["hazard"]][0]).to_numpy()
        b, v = base.to_numpy()[mask], val.to_numpy()[mask]
        out.append({**row, "baseline median (m/s)": round(float(np.nanmedian(b)), 1) if mask.any() else None,
                    "with validator median (m/s)": round(float(np.nanmedian(v)), 1) if mask.any() else None,
                    "reduced": f"{100 * float(np.nanmean(v < b)):.0f}%" if mask.any() else None})
    print("## Impact speed of missed hazards, ego relative to hazard, fused\n")
    _print(pl.DataFrame(out))


def occluded_vs_open() -> None:
    """does the parked van make the validator react earlier?"""
    df = hazard("main_fused_detail").filter(pl.col("family") == "ped")
    df = df.with_columns((pl.col("first_reaction_s") - pl.col("hazard_onset_s")).alias("reaction"))
    t = df.group_by("hazard").agg(
        pl.len().alias("cases"), pl.col("prevented").sum().alias("prevented"),
        pl.col("reaction").median().round(2).alias("first slowdown after onset, median (s)"),
        (pl.col("first_override_s") < pl.col("hazard_onset_s")).mean().round(3).alias("slowed before onset (share)"),
    ).sort("hazard")
    print("## Occluded vs. open pedestrians, fused\n")
    _print(t)


def braking_lead_crossing() -> None:
    """braking-lead cases crossing prediction loses."""
    keys = ["scene", "rollout", "hazard"]
    on = hazard("main_fused_detail").filter(pl.col("family") == "lead_brake")
    off = hazard("no_crossing").filter((pl.col("family") == "lead_brake") & (pl.col("sensor") == "fused"))
    j = on.select(*keys, "model", pl.col("prevented").alias("with"), pl.col("first_reaction_constraints").alias("with_first"),
                  pl.col("first_reaction_s").alias("with_t")).join(
        off.select(*keys, pl.col("prevented").alias("without"), pl.col("first_reaction_constraints").alias("without_first"),
                   pl.col("first_reaction_s").alias("without_t")), on=keys)
    lost, gained = j.filter(pl.col("without") & ~pl.col("with")), j.filter(~pl.col("without") & pl.col("with"))
    print(f"## Braking lead, with vs. without crossing prediction, fused: {lost.height} lost, {gained.height} gained\n")
    _print(lost.with_columns(pl.col("scene").str.slice(7, 8)).drop("rollout"))


TABLES = {"t1": table1, "crossing": crossing, "per_hazard": per_hazard, "baselines": baselines, "sweep": sweep,
          "tiers": tiers, "n1": n1_vs_n6, "mcnemar": mcnemar, "first_constraint": first_constraint,
          "decel": decelerations, "impact": impact_speeds, "occluded": occluded_vs_open,
          "braking_lead_crossing": braking_lead_crossing}


def main() -> None:
    chosen = sys.argv[1:] or list(TABLES)
    out = io.StringIO()
    for name in chosen:
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                TABLES[name]()
        except (FileNotFoundError, pl.exceptions.ComputeError) as missing:
            buffer.write(f"## {name}: waiting for its inputs ({str(missing).splitlines()[0][:120]})\n")
        print(buffer.getvalue())
        out.write(buffer.getvalue() + "\n")
    if not sys.argv[1:]:
        (FINAL / "tables.md").write_text(out.getvalue())


if __name__ == "__main__":
    main()
