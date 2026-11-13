"""Write the core and large scene lists from benchmark/scene_screen.csv.

Only scenes whose USDZ has LiDAR qualify (the safety validator needs it).
  core:  the --n-risk highest-risk scenes (screen_scenes.risk_score) plus
         --n-random others drawn with --seed, so the set has close-call traffic
         without being only that.
  large: --n-large scenes drawn with --seed, for the VaVAM runs.
  front: the --n-front scenes not in core with the highest front_risk, the
         crashes the ego causes itself (the filter cannot prevent being hit from
         behind): tight gaps to traffic ahead, cars passing close alongside and
         dense traffic.
  hard:  the --n-hard scenes with the highest front_risk of all, core included,
         for several rollouts each: the policy samples its plans, so the same
         scene can end differently.
Each list records how it was drawn in its header.

Usage (repo root): uv run python benchmark/select_scenes.py
"""

import argparse
from pathlib import Path

import polars as pl

SCREEN = Path("benchmark/scene_screen.csv")


def front_risk(screen: pl.DataFrame) -> pl.Expr:
    """Sum of how far each front margin falls inside a tight threshold, each in [0, 1]:
    front headway 1 s, front TTC 2 s, front gap 3 m, side clearance 0.75 m, and
    10 to 25 road users within 10 m."""

    def inside(column: str, tight: float) -> pl.Expr:
        return (1.0 - pl.col(column).clip(lower_bound=0.0) / tight).clip(0.0, 1.0)

    return (
        inside("min_front_headway_s", 1.0)
        + inside("min_front_ttc_s", 2.0)
        + inside("min_front_gap_m", 3.0)
        + inside("min_side_clearance_m", 0.75)
        + ((pl.col("n_close_road_users") - 10) / 15).clip(0.0, 1.0)
    )


def write_list(path: Path, header: str, scene_ids: list[str]) -> None:
    path.write_text(f"# {header}\n" + "\n".join(scene_ids) + "\n")
    print(f"Wrote {len(scene_ids)} scenes to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-risk", type=int, default=21)
    parser.add_argument("--n-random", type=int, default=21)
    parser.add_argument("--n-large", type=int, default=100)
    parser.add_argument("--n-front", type=int, default=40)
    parser.add_argument("--n-hard", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    screen = pl.read_csv(SCREEN).filter(pl.col("has_lidar"))
    print(f"{screen.height} screened scenes with LiDAR")

    by_risk = screen.sort(["risk", "scene_id"], descending=[True, False])
    risky = by_risk.head(args.n_risk)
    rest = by_risk.slice(args.n_risk).sample(args.n_random, seed=args.seed)
    write_list(
        Path("benchmark/scenes_core.txt"),
        f"{args.n_risk} highest-risk + {args.n_random} random (seed {args.seed}) "
        f"LiDAR scenes of {SCREEN}",
        sorted(risky["scene_id"].to_list() + rest["scene_id"].to_list()),
    )
    core = set(risky["scene_id"].to_list() + rest["scene_id"].to_list())
    front = (
        screen.filter(~pl.col("scene_id").is_in(list(core)))
        .with_columns(front_risk(screen).alias("front_risk"))
        .sort(["front_risk", "scene_id"], descending=[True, False])
        .head(args.n_front)
    )
    print(f"front_risk of the front set: {front['front_risk'].min():.2f} to {front['front_risk'].max():.2f}")
    write_list(
        Path("benchmark/scenes_front.txt"),
        f"{args.n_front} highest front_risk LiDAR scenes of {SCREEN} not in scenes_core.txt",
        sorted(front["scene_id"].to_list()),
    )
    hard = (
        screen.with_columns(front_risk(screen).alias("front_risk"))
        .sort(["front_risk", "scene_id"], descending=[True, False])
        .head(args.n_hard)
    )
    print(f"front_risk of the hard set: {hard['front_risk'].min():.2f} to {hard['front_risk'].max():.2f}")
    write_list(
        Path("benchmark/scenes_hard.txt"),
        f"{args.n_hard} highest front_risk LiDAR scenes of {SCREEN}",
        sorted(hard["scene_id"].to_list()),
    )
    write_list(
        Path("benchmark/scenes_large.txt"),
        f"{args.n_large} random (seed {args.seed}) LiDAR scenes of {SCREEN}",
        sorted(screen.sample(args.n_large, seed=args.seed)["scene_id"].to_list()),
    )


if __name__ == "__main__":
    main()
