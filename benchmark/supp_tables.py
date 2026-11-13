"""Supplementary tables from the final hazard test (benchmark/final/hazard_main.csv):
protocol counts (Sec. 2), prevention by hazard (Table 1), per-policy tables
(Tables 2-4), prevention by speed (Table 5), outcomes (Table 8), the rear-end and
miss analyses (Secs. 5-6) and the sensor-robustness sweep, when its CSVs exist.

Usage (repo root):  uv run python benchmark/supp_tables.py > benchmark/final/supp_tables.md
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

FINAL = Path("benchmark/final")
SENSORS = ("gt", "radar", "lidar", "fused")
HAZARDS = {
    "lead_brake_4": "Lead brakes, 4 m/s^2", "lead_brake_6": "Lead brakes, 6 m/s^2",
    "lead_brake_8": "Lead brakes, 8 m/s^2", "cut_in_10m": "Cut-in, 10 m", "cut_in_5m": "Cut-in, 5 m",
    "cut_in_10m_brake": "Cut-in, 10 m, then brakes", "ped_walk": "Pedestrian walks",
    "ped_walk_occluded": "  occluded by van", "ped_run": "Pedestrian runs", "ped_run_occluded": "  occluded by van",
}
FAMILY = pl.when(pl.col("hazard").str.starts_with("lead")).then(pl.lit("lead")).when(
    pl.col("hazard").str.starts_with("cut")).then(pl.lit("cut-in")).otherwise(pl.lit("pedestrian"))
REACT = "_react_1s"
ROBUST = {
    "clutter0": "radar clutter 0 / scan", "clutter3": "radar clutter 3 / scan",
    "pd095": "radar p_d 0.95", "pd080": "radar p_d 0.80",
    "dropout30": "LiDAR dropout 30%", "dropout60": "LiDAR dropout 60%",
}


def md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines)


def load(path: Path) -> pl.DataFrame:
    d = pl.read_csv(path, infer_schema_length=None)
    # "from behind" is only set when there is a new collision.
    d = d.with_columns(pl.col("^new_other_from_behind.*$").fill_null(False))
    return d.filter(pl.col("applicable")).with_columns(
        prevented=pl.col("baseline_hits_hazard") & ~pl.col("filtered_hits_hazard"),
        new_front_side=pl.col("new_other_collision") & ~pl.col("new_other_from_behind"),
        new_rear=pl.col("new_other_from_behind"),
        new_front_side_react=pl.col(f"new_other_collision{REACT}") & ~pl.col(f"new_other_from_behind{REACT}"),
        new_rear_react=pl.col(f"new_other_from_behind{REACT}"),
        family=FAMILY,
    )


def protocol(raw: pl.DataFrame) -> None:
    print("## Sec. 2: rollouts and applicability\n")
    per_model = raw.group_by("model").agg(
        rollouts=pl.struct("scene", "rollout").n_unique(), scenes=pl.col("scene").n_unique(),
    ).sort("model")
    print(md(per_model))
    cases = raw.unique(["model", "scene", "rollout", "hazard"])
    print(f"\n{cases.height} injections, {int(cases['applicable'].sum())} applicable; not applicable by hazard:\n")
    print(md(cases.filter(~pl.col("applicable")).group_by("model", "hazard").len().sort("model", "hazard")))
    print()


def by_hazard(d: pl.DataFrame) -> None:
    print("## Table 1: prevention by hazard (%, all policies pooled; n = baseline collisions)\n")
    hit = d.filter(pl.col("baseline_hits_hazard"))
    t = hit.group_by("hazard", "sensor").agg(n=pl.len(), p=(100 * pl.col("prevented").mean()).round(0).cast(pl.Int64))
    wide = t.pivot(on="sensor", index=["hazard", "n"], values="p")
    rows = [wide.filter(pl.col("hazard") == h).with_columns(hazard=pl.lit(label)) for h, label in HAZARDS.items()]
    print(md(pl.concat(rows).select("hazard", "n", *SENSORS)))
    print()


def per_policy(d: pl.DataFrame) -> None:
    for model in sorted(d["model"].unique()):
        m = d.filter(pl.col("model") == model)
        print(f"## Tables 2-4: {model} (injections, baseline collisions, prevented per sensing condition)\n")
        rows = []
        for h, label in HAZARDS.items():
            mh = m.filter(pl.col("hazard") == h)
            gt = mh.filter(pl.col("sensor") == "gt")
            row = {"hazard": label, "Inj.": gt.height, "Base": int(gt["baseline_hits_hazard"].sum())}
            for s in SENSORS:
                row[s] = int(mh.filter(pl.col("sensor") == s)["prevented"].sum())
            rows.append(row)
        gt = m.filter(pl.col("sensor") == "gt")
        total = {"hazard": "Total", "Inj.": gt.height, "Base": int(gt["baseline_hits_hazard"].sum())}
        extra = {k: {"hazard": label, "Inj.": None, "Base": None} for k, label in (
            ("new_front_side", "New front/side, recorded followers"), ("new_rear", "New rear-end, recorded followers"),
            ("new_front_side_react", "New front/side, followers react 1.0 s"),
            ("new_rear_react", "New rear-end, followers react 1.0 s"))}
        for s in SENSORS:
            ms = m.filter(pl.col("sensor") == s)
            total[s] = int(ms["prevented"].sum())
            for k in extra:
                extra[k][s] = int(ms[k].sum())
        print(md(pl.DataFrame(rows + [total, *extra.values()])))
        print()


def by_speed(d: pl.DataFrame) -> None:
    print("## Table 5: prevented (%) by ego speed at hazard onset (all policies; n = baseline collisions)\n")
    hit = d.filter(pl.col("baseline_hits_hazard")).with_columns(
        speed=pl.col("ego_speed_at_start_mps").cut([8, 12, 16], labels=["< 8", "8-12", "12-16", "> 16"], left_closed=True)  # noqa
    )
    t = hit.group_by("speed", "sensor").agg(n=pl.len(), p=(100 * pl.col("prevented").mean()).round(1))
    print(md(t.pivot(on="sensor", index=["speed", "n"], values="p").sort("speed").select("speed", "n", *SENSORS)))
    print()


def outcomes(d: pl.DataFrame) -> None:
    hit = d.filter(pl.col("baseline_hits_hazard"))
    for tag, fs, rear, label in (("", "new_front_side", "new_rear", "recorded followers (upper bound)"),
                                 ("react", "new_front_side_react", "new_rear_react", "followers react after 1.0 s")):
        print(f"## Table 8: outcomes over the {hit.filter(pl.col('sensor') == 'gt').height} baseline collisions, {label}\n")
        rows = []
        for name, expr in (("Hazard prevented", pl.col("prevented")),
                           ("Clean", pl.col("prevented") & ~pl.col(fs) & ~pl.col(rear)),
                           ("New front/side", pl.col(fs)), ("New rear-end", pl.col(rear))):
            rows.append({"outcome": name, **{s: int(hit.filter(pl.col("sensor") == s).select(expr.sum()).item()) for s in SENSORS}})
        print(md(pl.DataFrame(rows)))
        print()


def no_collision_cases(d: pl.DataFrame) -> None:
    ok = d.filter(~pl.col("baseline_hits_hazard") & (pl.col("sensor") == "fused"))
    print("## Sec. 5: applicable injections where the baseline does not collide (fused)\n")
    print(f"- cases: {ok.height}; validator intervenes: {int(ok['first_override_s'].is_not_null().sum())}; "
          f"new rear-end (recorded followers): {int(ok['new_rear'].sum())}; "
          f"new rear-end (followers react 1.0 s): {int(ok['new_rear_react'].sum())}; "
          f"new front/side: {int(ok['new_front_side'].sum())}\n")


def misses(d: pl.DataFrame) -> None:
    hit = d.filter(pl.col("baseline_hits_hazard"))
    key = ["model", "scene", "rollout", "hazard"]
    wide = hit.pivot(on="sensor", index=key, values="prevented")
    fused = hit.filter(pl.col("sensor") == "fused").join(wide.select(*key, pl.col("gt").alias("gt_prevented")), on=key)
    miss = fused.filter(~pl.col("prevented"))
    print("## Sec. 6: collisions not prevented with fusion\n")
    braking = miss.filter(~pl.col("gt_prevented"))
    sensing = miss.filter(pl.col("gt_prevented"))
    fam = lambda x: ", ".join(f"{r['family']} {r['len']}" for r in x.group_by("family").len().sort("family").iter_rows(named=True))
    print(f"- misses: {miss.height}; braking-limited (also missed with perfect perception): {braking.height} ({fam(braking)}); "
          f"sensing-limited: {sensing.height} ({fam(sensing)})")
    lead = lambda x: (x["baseline_hit_s"] - x["first_override_s"]).drop_nulls()
    acted = miss.filter(pl.col("first_override_s").is_not_null())
    prev = fused.filter(pl.col("prevented") & pl.col("first_override_s").is_not_null())
    q = lambda s: f"{s.median():.1f} s (IQR {s.quantile(0.25):.1f}-{s.quantile(0.75):.1f} s)"
    print(f"- intervened in {acted.height} of {miss.height} misses, first intervention {q(lead(acted))} before the "
          f"baseline impact; prevented cases {q(lead(prev))}")
    silent = miss.filter(pl.col("first_override_s").is_null())
    print(f"- no intervention: {silent.height} ({fam(silent)}); by hazard: "
          + ", ".join(f"{r['hazard']} {r['len']}" for r in silent.group_by("hazard").len().sort("hazard").iter_rows(named=True)))
    for other in ("lidar", "radar"):
        gain = int((wide["fused"] & ~wide[other]).sum())
        loss = int((~wide["fused"] & wide[other]).sum())
        print(f"- fusion vs {other} alone: prevents {gain} that {other} misses, loses {loss}")
    print(f"- fusion prevents {int((wide['fused'] & ~wide['lidar'] & ~wide['radar']).sum())} that neither sensor prevents alone\n")


def robustness(d: pl.DataFrame) -> None:
    rows = []
    def row(label: str, x: pl.DataFrame) -> dict:
        hit = x.filter(pl.col("baseline_hits_hazard"))
        p = hit.group_by("family").agg(pl.col("prevented").sum(), pl.len())
        fam = {r["family"]: f"{r['prevented']}/{r['len']}" for r in p.iter_rows(named=True)}
        return {"setting": label, "prevented": f"{int(hit['prevented'].sum())}/{hit.height} ({100 * hit['prevented'].mean():.1f}%)",
                "lead": fam.get("lead"), "cut-in": fam.get("cut-in"), "pedestrian": fam.get("pedestrian"),
                "new rear-end (react 1.0 s)": int(x["new_rear_react"].sum()), "intervened": int(x["first_override_s"].is_not_null().sum())}
    rows.append(row("default (clutter 1.5, p_d 0.9, no dropout)", d.filter(pl.col("sensor") == "fused")))
    for tag, label in ROBUST.items():
        path = FINAL / f"hazard_robust_{tag}.csv"
        if path.exists():
            x = load(path).filter(pl.col("sensor") == "fused")
            if x.height == d.filter(pl.col("sensor") == "fused").height:
                rows.append(row(label, x))
            else:
                rows.append({"setting": f"{label} (running: {x.height} cases so far)"})
    print("## Sensor robustness sweep (fused, all policies)\n")
    print(md(pl.DataFrame(rows, infer_schema_length=None)))
    print()


def main() -> None:
    raw = pl.read_csv(FINAL / "hazard_main.csv", infer_schema_length=None)
    d = load(FINAL / "hazard_main.csv")
    protocol(raw)
    by_hazard(d)
    per_policy(d)
    by_speed(d)
    outcomes(d)
    no_collision_cases(d)
    misses(d)
    robustness(d)


if __name__ == "__main__":
    main()
