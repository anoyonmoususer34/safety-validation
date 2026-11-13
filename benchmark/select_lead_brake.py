"""Write benchmark/scenes_leadbrake.txt: downloaded scenes where exp/scenario=lead_brake
finds a vehicle to brake early enough to matter, fastest recorded ego first.

Runs the runtime's own lead search (alpasim_runtime.lead_brake) on each scene's
recorded traffic.  The brake time is approximate: the runtime counts from the
closed-loop start, which this takes to be FIRST_BRAKE_S into the recording.

Usage (repo root): uv run python benchmark/select_lead_brake.py [--n 20]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import polars as pl
from alpasim_runtime.config import LeadBrakeConfig
from alpasim_runtime.lead_brake import _speed, apply_lead_brake
from alpasim_utils.artifact import Artifact

USDZ_DIR = Path("data/nre-artifacts/all-usdzs")
SCREEN = Path("benchmark/scene_screen.csv")
# Recording start to the earliest brake: egomotion context, force-GT warm-up
# (1.7 s) and LeadBrakeConfig.after_closed_loop_s, rounded up.
FIRST_BRAKE_S = 4.0


def _lead(usdz: Path) -> dict:
    artifact = Artifact(source=str(usdz))
    ego = artifact.rig.trajectory
    start, end = ego.time_range_us.start, ego.time_range_us.stop
    found = apply_lead_brake(
        artifact.traffic_objects, ego, start + int(FIRST_BRAKE_S * 1e6), end, LeadBrakeConfig()
    )
    if found is None:
        return {"uuid": usdz.stem, "lead": None, "brake_s": None, "ego_speed_mps": None}
    _, lead, brake_us = found
    return {
        "uuid": usdz.stem,
        "lead": lead,
        "brake_s": (brake_us - start) * 1e-6,
        "ego_speed_mps": _speed(ego, brake_us),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--latest-brake-s", type=float, default=12.0,
        help="leave the ego time to reach the stopped vehicle in a 20 s recording",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    screen = pl.read_csv(SCREEN).select("uuid", "scene_id")
    usdzs = [p for p in sorted(USDZ_DIR.glob("*.usdz")) if p.stem in set(screen["uuid"])]
    with ProcessPoolExecutor(args.workers) as pool:
        leads = pl.DataFrame(list(pool.map(_lead, usdzs)), infer_schema_length=None)
    leads = leads.join(screen, on="uuid").filter(
        pl.col("lead").is_not_null() & (pl.col("brake_s") <= args.latest_brake_s)
    )
    print(f"{leads.height} of {len(usdzs)} downloaded scenes have a lead vehicle to brake "
          f"by {args.latest_brake_s} s")
    chosen = leads.sort(["ego_speed_mps", "scene_id"], descending=[True, False]).head(args.n)
    with pl.Config(tbl_rows=-1, float_precision=1):
        print(chosen.select("scene_id", "lead", "brake_s", "ego_speed_mps"))
    Path("benchmark/scenes_leadbrake.txt").write_text(
        f"# {chosen.height} downloaded scenes with a lead vehicle for exp/scenario=lead_brake, "
        "fastest recorded ego first (benchmark/select_lead_brake.py)\n"
        + "\n".join(sorted(chosen["scene_id"].to_list()))
        + "\n"
    )
    print("Wrote benchmark/scenes_leadbrake.txt")


if __name__ == "__main__":
    main()
