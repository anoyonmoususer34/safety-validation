"""One table row from a finished hazard-test CSV (fused): prevented, per family,
new collisions with followers reacting after 1.0 s (all, and from behind).

Usage: uv run python benchmark/row_from_hazard.py CSV [--alpamayo]
"""
import sys

import polars as pl

path, alpamayo = sys.argv[1], "--alpamayo" in sys.argv
d = pl.read_csv(path, infer_schema_length=None).filter(pl.col("applicable") & (pl.col("sensor") == "fused"))
if alpamayo:
    d = d.filter(pl.col("model") == "Alpamayo 1.5")
d = d.with_columns(prevented=pl.col("baseline_hits_hazard") & ~pl.col("filtered_hits_hazard"),
                   family=pl.col("hazard").str.replace(r"^(lead_brake|cut_in|ped)_.*$", "$1"))
hit = d.filter(pl.col("baseline_hits_hazard"))
fam = {r["family"]: f"{r['p']}/{r['n']}" for r in hit.group_by("family").agg(p=pl.col("prevented").sum(), n=pl.len()).iter_rows(named=True)}
new = int(d["new_other_collision_react_1s"].sum())
rear = int((d["new_other_collision_react_1s"] & d["new_other_from_behind_react_1s"].fill_null(False)).sum())
print(f"{path}: cases {d.height}, prevented {int(hit['prevented'].sum())}/{hit.height} "
      f"({100 * hit['prevented'].mean():.0f}%), lead {fam.get('lead_brake')}, cut-in {fam.get('cut_in')}, "
      f"ped {fam.get('ped')}, new (react 1 s) {new}, of which rear {rear}")
