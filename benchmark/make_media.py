"""Figures and videos for the paper and its supplementary.

Camera material comes from the closed-loop braking-car runs, whose evaluation
videos stack a bird's-eye view, a metrics table (with run IDs) and the front
camera; only the camera tile is used.  Video frame k is evaluation step k.

Usage (repo root):  uv run python benchmark/make_media.py teaser b2 b3 b4 b5
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import math
import sys

from alpasim_utils.geometry import pose_from_grpc
from alpasim_utils.logs import async_read_pb_log

sys.path.insert(0, str(Path(__file__).parent))
import counterfactual_braking as cf  # noqa: E402
import hazard_stress as hs  # noqa: E402
import replay_validator as rv  # noqa: E402

OUT = Path("benchmark/media")
SHORT_RUNS = Path("benchmark/runs_leadbrake_short")
# The front-camera tile of an evaluation video (rows, columns), below its label
# box ("Command", "Safety sensors": up to row 565) and cropped to 16:9.
CAMERA_TILE = (slice(570, 991), slice(76, 824))
TEASER_SIZE = (1280, 720)
DPI = 200


@dataclass(frozen=True)
class Rollout:
    """One finished closed-loop rollout of a scene."""

    run: str
    scene: str  # 8-character clip prefix
    runs_dir: Path = SHORT_RUNS
    rollout: str = ""  # rollout prefix, when the run has several of the scene

    @property
    def dir(self) -> Path:
        return Path(sorted(glob.glob(str(self.runs_dir / self.run / "rollouts" / f"clipgt-{self.scene}*" / f"{self.rollout}*")))[0])

    @property
    def video(self) -> Path:
        return Path(glob.glob(str(self.dir / "*camera_front*.mp4"))[0])

    def step_times(self) -> list[int]:
        """Evaluation timestamps (us), one per video frame."""
        df = pl.read_parquet(self.dir / "metrics.parquet")
        return sorted(df["timestamps_us"].unique().to_list())

    def first_collision_step(self) -> int | None:
        df = pl.read_parquet(self.dir / "metrics.parquet")
        hits = df.filter(pl.col("name").str.starts_with("collision") & (pl.col("values") > 0))
        return None if hits.is_empty() else self.step_times().index(int(hits["timestamps_us"].min()))

    def decisions(self) -> list[tuple[int, dict]]:
        """(request time, the safety validator's debug record) per Drive call."""

        async def read() -> list[tuple[int, dict]]:
            out, request_us = [], None
            async for entry in async_read_pb_log(str(self.dir / "rollout.asl")):
                kind = entry.WhichOneof("log_entry")
                if kind == "driver_request":
                    request_us = entry.driver_request.time_now_us
                elif kind == "driver_return":
                    debug = pickle.loads(entry.driver_return.debug_info.unstructured_debug_info)
                    if debug.get("safety_validator"):
                        out.append((request_us, debug["safety_validator"]))
            return out

        return asyncio.run(read())

    def camera_frame(self, step: int) -> np.ndarray:
        """The front-camera tile at evaluation ``step``, RGB."""
        capture = cv2.VideoCapture(str(self.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, step)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"{self.video} has no frame {step}")
        return cv2.cvtColor(frame[CAMERA_TILE], cv2.COLOR_BGR2RGB)


def _lead_brake_us(run: str, scene: str) -> int:
    """When the lead vehicle starts braking, from the runtime log."""
    for log in glob.glob(str(SHORT_RUNS / run / "txt-logs" / "*.log")):
        for line in open(log):
            if "lead_brake:" in line and scene in line:
                return int(line.rsplit("brakes at", 1)[1].split()[0])
    raise ValueError(f"no lead_brake line for {scene} in {run}")


def _proposed_speed(decision: dict) -> float:
    return decision["validator_result"]["trace_data"]["inputs"]["proposed_action"]["target_speed"]


def _save_png(rgb: np.ndarray, path: Path, size: tuple[int, int] = TEASER_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR))
    print(f"wrote {path}")


def teaser(scene: str = "6250146a") -> None:
    """Teaser (A1) and B1: the braking car, baseline vs. validator, closed loop."""
    baseline = Rollout("alpamayo1_5_leadbrake__s2", scene)
    validator = Rollout("alpamayo1_5_safety_fused_leadbrake__s2", scene)
    times = baseline.step_times()
    assert times == validator.step_times(), "rollouts must share evaluation steps"
    brake_step = int(np.searchsorted(times, _lead_brake_us(baseline.run, scene)))
    crash_step = baseline.first_collision_step()
    t0 = times[0]
    frames = {
        "teaser_baseline_before": baseline.camera_frame(brake_step),
        "teaser_baseline_crash": baseline.camera_frame(crash_step),
        "teaser_validator_stop": validator.camera_frame(crash_step),
    }
    for name, rgb in frames.items():
        _save_png(rgb, OUT / "fig" / f"{name}.png")

    # B1: the same frames with the validator's first slowdown as a trace line.
    overrides = [(t, d) for t, d in validator.decisions() if d["reason"] == "speed_override"]
    t_first, first = overrides[0]
    trace = (
        f"Validator at t = {(t_first - t0) * 1e-6:.1f} s: "
        f"{', '.join(v.replace('_VIOLATION', '').replace('_', ' ').lower() for v in first['violations'])}; "
        f"proposed {_proposed_speed(first):.1f} m/s, safe {first['safe_speed']:.1f} m/s"
    )
    captions = [
        f"(a) Baseline, lead brakes (t = {(times[brake_step] - t0) * 1e-6:.1f} s)",
        f"(b) Baseline, impact (t = {(times[crash_step] - t0) * 1e-6:.1f} s)",
        f"(c) With validator, same time (t = {(times[crash_step] - t0) * 1e-6:.1f} s)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), dpi=DPI)
    for ax, rgb, caption in zip(axes, frames.values(), captions):
        ax.imshow(rgb)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")
    fig.text(0.5, 0.02, trace, ha="center", fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUT / "fig" / "B1_braking_car.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}\n{trace}")


# ---------------------------------------------------------------------------
# Bird's-eye views of hazard-test cases (B2-B5).  The hazards are injected into
# the logged traffic and not rendered, so these are drawn from the test's state.

SOURCE_COLORS = {"lidar": "#2a9d8f", "radar": "#e76f51", "lidar+radar": "#7b2cbf", "gt": "#264653"}
LANE_WIDTH_M = 3.5


@dataclass(frozen=True)
class HazardCase:
    scene: str  # 8-character clip prefix
    rollout: str  # 8-character rollout prefix
    hazard: str


@dataclass
class Simulated:
    log: rv.RolloutLog
    hazard_id: str
    start_us: int
    result: dict  # hazard_stress.closed_loop's
    baseline: list  # the logged ego's track, counterfactual_braking.counterfactual_track


def _asl(case: HazardCase) -> Path:
    paths = glob.glob(f"benchmark/runs*/*/rollouts/clipgt-{case.scene}*/{case.rollout}*/rollout.asl")
    baseline = [p for p in paths if "safety" not in p and "leadbrake" not in p]
    return Path(sorted(baseline)[0])


def simulate(case: HazardCase, sensor: str, settings: list[str] = ()) -> Simulated:
    log = asyncio.run(rv._read_log(_asl(case)))
    mod, hazard_id, start_us = hs.inject(log, hs.HAZARDS[case.hazard])
    result = hs.closed_loop(
        mod, hazard_id, start_us, sensor, rv._load_radars(rv.DEFAULT_RADAR_CONFIG),
        rv.validator_config(list(settings)), single_candidate=False,
    )
    baseline, _ = cf.counterfactual_track(mod, {})
    return Simulated(mod, hazard_id, start_us, result, baseline)


def _hazard_onset_us(sim: Simulated, case: HazardCase) -> int:
    return hs.hazard_onset_us(hs.HAZARDS[case.hazard], sim.start_us)


class View:
    """A fixed frame on the scene: ``origin`` on the ego's logged path, x along it."""

    def __init__(self, path: cf.LoggedPath, s: float) -> None:
        xy, yaw, _ = path.at(s)
        self.origin, self.yaw, self.path = xy, yaw, path
        c, si = math.cos(yaw), math.sin(yaw)
        self._rot = np.array([[c, si], [-si, c]])

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy) - self.origin) @ self._rot.T

    def angle(self, yaw: float) -> float:
        return yaw - self.yaw


def _box(ax, view: View, xy, yaw, length, width, **style) -> None:
    corners = view(cf._box_corners(np.asarray(xy, dtype=float), yaw, length, width))
    ax.add_patch(matplotlib.patches.Polygon(corners, closed=True, **style))


def _at(track: list, time_us: int) -> tuple:
    return min(track, key=lambda step: abs(step[0] - time_us))


def bev_panel(ax, sim: Simulated, time_us: int, view: View, window: tuple, title: str) -> None:
    """One snapshot: lane lines from the logged path, road users, LiDAR returns,
    detected obstacles by source, the ego and (dashed) the baseline ego."""
    (x0, x1), (y0, y1) = window
    s = np.linspace(view.path.arc[0], view.path.arc[-1], 400)
    centre = np.array([view.path.at(v)[0] for v in s])
    heading = np.array([view.path.at(v)[1] for v in s])
    normal = np.column_stack([-np.sin(heading), np.cos(heading)])
    for offset, style in ((0.0, dict(ls=(0, (6, 6)), lw=0.8)), (LANE_WIDTH_M / 2, dict(lw=1.0)),
                          (-LANE_WIDTH_M / 2, dict(lw=1.0))):
        line = view(centre + offset * normal)
        ax.plot(line[:, 0], line[:, 1], color="#9aa0a6", zorder=1, **style)

    t, ego_xy, ego_yaw, ego_v, _ = _at(sim.result["track"], time_us)
    decisions = [d for d in sim.result["decisions"] if d["time_us"] <= t]
    decision = decisions[-1] if decisions else None
    # Road users at their logged places.
    for actor_id, box in cf._actor_boxes(sim.log, t):
        se3 = pose_from_grpc(sim.log.actor_poses[actor_id][t]).as_se3()
        size = sim.log.aabbs[actor_id]
        if actor_id == sim.hazard_id:
            style = dict(facecolor="#d62828", edgecolor="#7f1010", lw=1.2, zorder=4)
            ax.add_patch(matplotlib.patches.Circle(view(se3[:2, 3]), 1.6, fill=False, edgecolor="#d62828", lw=1.0, zorder=4))
        elif actor_id == hs.OCCLUDER_ID:
            style = dict(facecolor="#495057", edgecolor="#212529", lw=1.0, zorder=3)
        else:
            style = dict(facecolor="#dee2e6", edgecolor="#868e96", lw=0.8, zorder=2)
        _box(ax, view, se3[:2, 3], cf._yaw(se3), size.x, size.y, **style)
    # LiDAR returns, as the simulated sensor sees them from this ego.
    if decision is not None:
        rig = decision["rig_se3"]
        points = rv._lidar_spin_points(
            rv._traffic(sim.log), decision["time_us"], np.linalg.inv(rig), np.random.default_rng(0)
        )
        if len(points):
            world = (points.astype(float) + rv.LIDAR_IN_RIG) @ rig[:3, :3].T + rig[:3, 3]
            p = view(world[:, :2])
            ax.scatter(p[:, 0], p[:, 1], s=2.5, color="#868e96", zorder=2, linewidths=0)
        for obstacle in decision["obstacles"]:
            centre_xy = (rig @ np.array([obstacle["x"], obstacle["y"], 0.0, 1.0]))[:2]
            color = SOURCE_COLORS[obstacle["source"]]
            half_l, half_w = obstacle.get("half_length_m", 0.0), obstacle.get("half_width_m", 0.0)
            if half_l > 0.05 and half_w > 0.05:
                _box(ax, view, centre_xy, math.atan2(rig[1, 0], rig[0, 0]), 2 * half_l, 2 * half_w,
                     fill=False, edgecolor=color, lw=1.4, zorder=5)
            else:
                q = view(centre_xy)
                ax.plot(q[0], q[1], marker="x", color=color, ms=6, mew=1.6, zorder=5)
    size = sim.log.aabbs[hs.EGO_ID]
    _, base_xy, base_yaw, _, _ = _at(sim.baseline, t)
    _box(ax, view, base_xy, base_yaw, size.x, size.y, fill=False, edgecolor="#1d3557", lw=1.0, ls="--", zorder=5)
    _box(ax, view, ego_xy, ego_yaw, size.x, size.y, facecolor="#457b9d", edgecolor="#1d3557", lw=1.2, zorder=6)

    commanded = "none"
    constraint = "none"
    if decision is not None and decision["reason"] == "speed_override":
        commanded = f"{decision['safe_speed']:.1f} m/s"
        constraint = ", ".join(v.replace("_VIOLATION", "").replace("_", " ").lower() for v in decision["violations"])
    t_rel = (t - min(sim.log.actor_poses[hs.EGO_ID])) * 1e-6
    ax.set_title(f"{title}\nt = {t_rel:.1f} s, ego {ego_v:.1f} m/s, commanded {commanded}\nconstraint: {constraint}", fontsize=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(fig, note: str) -> None:
    handles = [
        matplotlib.patches.Patch(facecolor="#457b9d", edgecolor="#1d3557", label="ego"),
        matplotlib.patches.Patch(fill=False, edgecolor="#1d3557", ls="--", label="baseline ego"),
        matplotlib.patches.Patch(facecolor="#d62828", label="hazard"),
        matplotlib.patches.Patch(facecolor="#495057", label="parked van"),
        matplotlib.patches.Patch(facecolor="#dee2e6", edgecolor="#868e96", label="other road users"),
        matplotlib.lines.Line2D([], [], marker=".", ls="", color="#adb5bd", label="LiDAR returns"),
        *[matplotlib.patches.Patch(fill=False, edgecolor=c, label=f"detected: {k}") for k, c in SOURCE_COLORS.items()],
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=7, frameon=False,
               title=note, title_fontsize=7)


def bev_grid(rows: list[tuple[str, Simulated]], case: HazardCase, times: list[tuple[str, int]],
             window: tuple, centre_s: float | None, path: Path, note: str) -> None:
    """Rows of snapshots.  With ``centre_s`` the view is fixed on the scene there;
    without, each panel is centred on the hazard at that moment."""
    (x0, x1), (y0, y1) = window
    panel_w = 4.2
    fig, axes = plt.subplots(
        len(rows), len(times), figsize=(panel_w * len(times), (panel_w * (y1 - y0) / (x1 - x0) + 0.62) * len(rows) + 0.6),
        dpi=DPI, squeeze=False, layout="constrained",
    )
    for r, (label, sim) in enumerate(rows):
        ego_path = cf.LoggedPath(sim.log.actor_poses[hs.EGO_ID])
        for c, (moment, time_us) in enumerate(times):
            hit_us = sim.result["hazard_hit_us"]
            if hit_us is not None and time_us > hit_us:
                moment, time_us = "impact", hit_us
            s = centre_s if centre_s is not None else _hazard_s(sim, ego_path, time_us)
            bev_panel(axes[r][c], sim, time_us, View(ego_path, s), window, f"{label}: {moment}")
    _legend(fig, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


LANE_NOTE = "Lane lines drawn from the logged ego path (3.5 m lanes). Hazards are injected, not rendered."


def _hazard_s(sim: Simulated, path: cf.LoggedPath, time_us: int) -> float:
    """Distance along the ego's logged path nearest the hazard at ``time_us``."""
    poses = sim.log.actor_poses[sim.hazard_id]
    hazard_xy = pose_from_grpc(poses[min(poses, key=lambda u: abs(u - time_us))]).as_se3()[:2, 3]
    return float(path.arc[np.argmin(np.linalg.norm(path.xy - hazard_xy, axis=1))])


def _crossing_s(sim: Simulated, case: HazardCase) -> float:
    """Distance along the ego's logged path where the hazard meets it."""
    path = cf.LoggedPath(sim.log.actor_poses[hs.EGO_ID])
    hazard_xy = pose_from_grpc(sim.log.actor_poses[sim.hazard_id][min(
        sim.log.actor_poses[sim.hazard_id], key=lambda u: abs(u - sim.start_us))]).as_se3()[:2, 3]
    return float(path.arc[np.argmin(np.linalg.norm(path.xy - hazard_xy, axis=1))])


def b2_occluded_pedestrian() -> None:
    case = HazardCase("d55c0611", "89b0de0c", "ped_walk_occluded")
    fused, radar = simulate(case, "fused"), simulate(case, "radar")
    onset = _hazard_onset_us(fused, case)
    times = [("1 s before stepping out", onset - 1_000_000), ("1 s after stepping out", onset + 1_000_000),
             ("pedestrian reaches the lane centre", fused.start_us)]
    s = _crossing_s(fused, case)
    bev_grid([("Radar only", radar), ("LiDAR + radar", fused)], case, times,
             ((-45, 12), (-9, 9)), s, OUT / "fig" / "B2_occluded_pedestrian.png", LANE_NOTE)


def b3_cut_in() -> None:
    case = HazardCase("5c1b7555", "b7cf474a", "cut_in_5m")
    with_c, without_c = simulate(case, "fused"), simulate(case, "fused", ["crossing_horizon_s=0"])
    start = with_c.start_us
    times = [("cut-in starts", start), ("1 s later", start + 1_000_000), ("2 s later", start + 2_000_000)]
    bev_grid([("Without crossing prediction", without_c), ("With crossing prediction", with_c)], case, times,
             ((-32, 18), (-7, 8)), None, OUT / "fig" / "B3_cut_in.png", LANE_NOTE + " Each view is centred on the cut-in car.")


def _speed_plot(ax, sim: Simulated, case: HazardCase, label: str) -> None:
    t0 = min(sim.log.actor_poses[hs.EGO_ID])
    rel = lambda us: (us - t0) * 1e-6  # noqa: E731
    ax.plot([rel(s[0]) for s in sim.baseline], [s[3] for s in sim.baseline], color="#adb5bd", lw=2, label="baseline (logged)")
    ax.plot([rel(s[0]) for s in sim.result["track"]], [s[3] for s in sim.result["track"]], color="#1d3557", lw=2, label=label)
    onset = _hazard_onset_us(sim, case)
    ax.axvline(rel(onset), color="#d62828", ls="--", lw=1, label="hazard onset")
    overrides = [d["time_us"] for d in sim.result["decisions"] if d["reason"] == "speed_override" and d["time_us"] >= onset]
    if overrides:
        ax.axvline(rel(overrides[0]), color="#2a9d8f", ls="--", lw=1, label="first intervention after onset")
    from_hazard = hs._only(sim.log, {sim.hazard_id})
    hit = cf.first_collision(from_hazard, None, sim.baseline, {})
    if hit is not None:
        ax.axvline(rel(hit[0]), color="#000000", ls=":", lw=1.2, label="baseline impact")
    if sim.result["hazard_hit_us"] is not None:
        hit_us = sim.result["hazard_hit_us"]
        hit_v = _at(sim.result["track"], hit_us)[3]
        base_v = _at(sim.baseline, hit[0])[3] if hit is not None else float("nan")
        ax.axvline(rel(hit_us), color="#e76f51", ls=":", lw=1.2,
                   label=f"{label}: impact at {hit_v:.0f} m/s (baseline {base_v:.0f} m/s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ego speed (m/s)")
    ax.set_xlim(rel(onset) - 4, rel(onset) + 6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def b4_braking_limited_miss() -> None:
    case = HazardCase("05f35348", "ae06703c", "ped_run")
    gt = simulate(case, "gt")
    onset = _hazard_onset_us(gt, case)
    times = [("1 s before stepping out", onset - 1_000_000), ("0.5 s after stepping out", onset + 500_000),
             ("0.2 s before impact", gt.result["hazard_hit_us"] - 200_000)]
    bev_grid([("Perfect sensing (gt)", gt)], case, times, ((-85, 15), (-10, 10)), _crossing_s(gt, case),
             OUT / "fig" / "B4_braking_limited_miss_bev.png",
             LANE_NOTE + " A runner at 3 m/s gives about 1 s of warning; stopping from 34 m/s needs 5.7 s at 6 m/s^2.")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=DPI)
    _speed_plot(ax, gt, case, "validator (gt)")
    # Below the axes: inside, it covers the braking curve.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    ax.set_title("Running pedestrian at 34 m/s: braking-limited miss", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig" / "B4_braking_limited_miss_speed.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig' / 'B4_braking_limited_miss_speed.png'}")


def b5_speed_vs_time() -> None:
    case = HazardCase("6250146a", "7ab1c9a4", "lead_brake_6")
    fused = simulate(case, "fused")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=DPI)
    _speed_plot(ax, fused, case, "validator (LiDAR + radar)")
    ax.set_title("Lead vehicle brakes at 6 m/s^2: prevented", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig" / "B5_speed_vs_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _sense_window(log: rv.RolloutLog, sensor_mode: str, effects, until_us: int, seconds: float = 1.5):
    """Run a fresh filter's sensing over the Drive calls in the ``seconds`` up to
    ``until_us``, as the replay does; the last call's obstacles and its time."""
    from alpasim_driver.safety_filter import SafetyFilter

    safety_filter = SafetyFilter(rv.validator_config([]), rv.OUTPUT_FREQUENCY_HZ)
    traffic, ego_trajectory = rv._traffic(log), rv._trajectory(log.ego_poses)
    last = None
    for drive in log.drives:
        t = drive.selected.poses[0].timestamp_us
        if not until_us - seconds * 1e6 <= t <= until_us or t not in log.ego_speed:
            continue
        rig_from_local = np.linalg.inv(pose_from_grpc(drive.selected.poses[0].pose).as_se3().astype(np.float64))
        sensed = rv.sense(safety_filter, sensor_mode, traffic, ego_trajectory, t, rig_from_local,
                          log.ego_speed[t], log.ego_yaw_rate[t], [] if sensor_mode == "lidar" else rv._load_radars(rv.DEFAULT_RADAR_CONFIG),
                          seed=0, effects=effects, mode=rv.FIXED_MODES[sensor_mode])
        last = (t, rig_from_local, sensed.obstacles)
    return last


def _lidar_points(log, t, rig_from_local, effects) -> np.ndarray:
    """(N, 2) rig-frame LiDAR returns, degraded as rv.sense degrades them."""
    from alpasim_utils.lidar_payload import unpack_lidar_spin

    points = rv._lidar_spin_points(rv._traffic(log), t, rig_from_local, np.random.default_rng([0, t, 1]))
    payload = rv.pack_lidar_spin(points.tobytes(), len(points), t, rv.RIG_TO_LIDAR)
    if effects is not None:
        payload = rv.degrade_lidar_payload(payload, effects, np.random.default_rng([0, t, 2]))
    xyz = unpack_lidar_spin(payload).xyz.astype(float) + rv.LIDAR_IN_RIG
    return xyz[:, :2]


def _radar_points(log, t, effects) -> np.ndarray:
    scan = rv.simulate_radar_scan(rv._load_radars(rv.DEFAULT_RADAR_CONFIG), rv._trajectory(log.ego_poses),
                                  rv._traffic(log), t, np.random.default_rng([0, t]))
    if effects is not None:
        scan = rv.degrade_radar(scan, effects, np.random.default_rng([0, t, 3]))
    return np.asarray(scan.xy)


def _rig_panel(ax, log, t, rig_from_local, obstacles, points, point_color, title, window) -> None:
    """A rig-frame view: x forward (right), y left (up)."""
    for actor_id, poses in log.actor_poses.items():
        if actor_id == hs.EGO_ID or t not in poses:
            continue
        se3 = rig_from_local @ pose_from_grpc(poses[t]).as_se3().astype(np.float64)
        size = log.aabbs[actor_id]
        corners = cf._box_corners(se3[:2, 3], cf._yaw(se3), size.x, size.y)
        ax.add_patch(matplotlib.patches.Polygon(corners, closed=True, facecolor="#dee2e6", edgecolor="#868e96", lw=0.8, zorder=2))
    ego = log.aabbs[hs.EGO_ID]
    ax.add_patch(matplotlib.patches.Polygon(cf._box_corners(np.array([1.3, 0.0]), 0.0, ego.x, ego.y), closed=True,
                                            facecolor="#457b9d", edgecolor="#1d3557", zorder=6))
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=2.5, color=point_color, zorder=3, linewidths=0)
    for obstacle in obstacles:
        color = SOURCE_COLORS[obstacle["source"]]
        half_l, half_w = obstacle.get("half_length_m", 0.0), obstacle.get("half_width_m", 0.0)
        if half_l > 0.05 and half_w > 0.05:
            ax.add_patch(matplotlib.patches.Rectangle((obstacle["x"] - half_l, obstacle["y"] - half_w), 2 * half_l, 2 * half_w,
                                                      fill=False, edgecolor=color, lw=1.4, zorder=5))
        else:
            ax.plot(obstacle["x"], obstacle["y"], marker="x", color=color, ms=6, mew=1.6, zorder=5)
    (x0, x1), (y0, y1) = window
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def b8_rain_scene(scene: str = "5c1b7555", rollout: str = "602d7e26", step: int = 60) -> None:
    """Camera frame of the rain scene, and the simulated LiDAR and radar in clear
    weather and with the runtime's rain model, with what the filter detects."""
    run = Rollout("alpamayo1_5", scene, Path("benchmark/runs"), rollout)
    t = run.step_times()[step]
    log = asyncio.run(rv._read_log(run.dir / "rollout.asl"))
    rain = rv.WEATHER_EFFECTS["rain"]
    window = ((-10, 70), (-14, 14))
    panels = [
        ("LiDAR, clear weather", "lidar", None, "#868e96"),
        ("LiDAR, rain model", "lidar", rain, "#868e96"),
        ("Radar, rain model", "radar", rain, "#e76f51"),
    ]
    fig = plt.figure(figsize=(14, 2.9), dpi=DPI, layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1])
    ax = fig.add_subplot(grid[0])
    ax.imshow(run.camera_frame(step))
    ax.axis("off")
    ax.set_title(f"Front camera, rain (t = {(t - run.step_times()[0]) * 1e-6:.1f} s)", fontsize=8)
    for i, (title, mode, effects, color) in enumerate(panels):
        t_sense, rig_from_local, obstacles = _sense_window(log, mode, effects, t)
        points = _lidar_points(log, t_sense, rig_from_local, effects) if mode == "lidar" else _radar_points(log, t_sense, effects)
        _rig_panel(fig.add_subplot(grid[i + 1]), log, t_sense, rig_from_local, obstacles, points, color,
                   f"{title}: {len(obstacles)} obstacle{'' if len(obstacles) == 1 else 's'}", window)
    handles = [
        matplotlib.patches.Patch(facecolor="#457b9d", edgecolor="#1d3557", label="ego"),
        matplotlib.patches.Patch(facecolor="#dee2e6", edgecolor="#868e96", label="road users (true boxes)"),
        matplotlib.lines.Line2D([], [], marker=".", ls="", color="#868e96", label="LiDAR returns"),
        matplotlib.lines.Line2D([], [], marker=".", ls="", color="#e76f51", label="radar detections"),
        matplotlib.patches.Patch(fill=False, edgecolor=SOURCE_COLORS["lidar"], label="obstacle: LiDAR"),
        matplotlib.lines.Line2D([], [], marker="x", ls="", color=SOURCE_COLORS["radar"], label="obstacle: radar"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=7, frameon=False,
               title="Simulated sensors with the runtime's rain model; the rendered sensor data is not logged.", title_fontsize=7)
    path = OUT / "fig" / "B8_rain_scene.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Videos (section C).  H.264, 10 fps (one frame per simulation step).

FPS = 10
STRIP_H = 70  # status strip under a camera tile, px


def _write_video(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (w - w % 2, h - h % 2), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "veryfast"], macro_block_size=1,
    )
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame[: h - h % 2, : w - w % 2]))
    writer.close()
    print(f"wrote {path} ({len(frames) / fps:.1f} s)")


def _label(rgb: np.ndarray, text: str, origin=(14, 34), scale=0.9) -> np.ndarray:
    out = rgb.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 10), (x + tw + 8, y + 10), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _status_strip(width: int, text: str, intervening: bool) -> np.ndarray:
    """A dark strip with a state dot and ``text``, wrapped onto two lines if needed."""
    strip = np.full((STRIP_H, width, 3), (33, 37, 41), dtype=np.uint8)
    color = (244, 162, 97) if intervening else (82, 183, 136)
    cv2.circle(strip, (24, STRIP_H // 2), 10, color, -1, cv2.LINE_AA)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    lines, line = [], ""
    for word in text.split(" "):
        candidate = f"{line} {word}".strip()
        if line and cv2.getTextSize(candidate, font, scale, 1)[0][0] > width - 60:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    top = STRIP_H // 2 + 6 - 11 * (len(lines) - 1)
    for i, text_line in enumerate(lines[:2]):
        cv2.putText(strip, text_line, (46, top + 24 * i), font, scale, (240, 240, 240), 1, cv2.LINE_AA)
    return strip


def _constraints(violations: list[str]) -> str:
    """Readable names of the violated constraints, as shown in the videos (the
    rear check is documented separately and not shown)."""
    names = [v.replace("_VIOLATION", "").replace("_", " ").lower() for v in violations if v != "REAR_COLLISION_RISK"]
    return f" ({', '.join(names)})" if names else ""


def _decision_text(decision: dict | None) -> tuple[str, bool]:
    """What the validator does at this step.  The driver applies its speed only
    when that is below the plan's (``_brake_along_path``); otherwise the policy's
    own plan, already slower, is driven unchanged."""
    if decision is None or decision["reason"] != "speed_override":
        return "Validator: OK (model plan kept)", False
    constraint = _constraints(decision["violations"])
    proposed, safe = decision.get("proposed"), decision["safe_speed"]
    if proposed is None:
        return f"Validator: braking{constraint} to {safe:.1f} m/s", True
    if safe < proposed:
        return f"Validator: braking{constraint}: {proposed:.1f} -> {safe:.1f} m/s", True
    return f"Validator: hazard flagged{constraint}; model plan already slowing to {proposed:.1f} m/s, kept", True


def _logged_decisions(rollout: Rollout) -> list[tuple[int, dict]]:
    """The validator's decisions in a closed-loop rollout, with the proposed speed."""
    out = []
    for t, d in rollout.decisions():
        out.append((t, {**d, "proposed": _proposed_speed(d) if d.get("validator_result") else None}))
    return out


def _decision_at(decisions: list[tuple[int, dict]], time_us: int) -> dict | None:
    before = [d for t, d in decisions if t <= time_us]
    return before[-1] if before else None


# The braking-car video ends at this step (8.3 s): afterwards recorded traffic,
# which does not react, drives through the validator's stopped ego and the
# renderer smears those cars over its camera.  The last frame is then held.
V1_LAST_STEP = 83
V1_END_HOLD_S = 1.6


def v1_braking_car(scene: str = "6250146a") -> None:
    baseline = Rollout("alpamayo1_5_leadbrake__s2", scene)
    validator = Rollout("alpamayo1_5_safety_fused_leadbrake__s2", scene)
    times = baseline.step_times()
    decisions = _logged_decisions(validator)
    crash = {r: r.first_collision_step() for r in (baseline, validator)}
    capture = {r: cv2.VideoCapture(str(r.video)) for r in (baseline, validator)}
    frames = []
    for k, t in enumerate(times[: V1_LAST_STEP + 1]):
        tiles = []
        for r, label in ((baseline, "Baseline"), (validator, "With validator (LiDAR + radar)")):
            ok, frame = capture[r].read()
            rgb = cv2.cvtColor(frame[CAMERA_TILE], cv2.COLOR_BGR2RGB)
            # Only the baseline's impact is labelled.
            impact = "   IMPACT" if r is baseline and crash[r] is not None and k >= crash[r] else ""
            tiles.append(_label(rgb, f"{label}   t = {(t - times[0]) * 1e-6:4.1f} s{impact}", scale=0.6, origin=(12, 28)))
        text, active = _decision_text(_decision_at(decisions, t))
        row = np.hstack([tiles[0], np.full((tiles[0].shape[0], 6, 3), 255, np.uint8), tiles[1]])
        frames.append(np.vstack([row, _status_strip(row.shape[1], text, active)]))
    frames += [frames[-1]] * int(V1_END_HOLD_S * FPS)
    _write_video(frames, OUT / "videos" / "V1_braking_car.mp4")


def _figure_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _bev_video(rows: list[tuple[str, Simulated]], start_us: int, end_us: int, window: tuple,
               centre: float | None, path: Path, note: str) -> None:
    """Side-by-side bird's-eye views, one frame per step."""
    (x0, x1), (y0, y1) = window
    panel_w = 6.4
    frames = []
    times = [t for t, *_ in rows[0][1].baseline if start_us <= t <= end_us]
    for t in times:
        fig, axes = plt.subplots(1, len(rows), figsize=(panel_w * len(rows), panel_w * (y1 - y0) / (x1 - x0) + 1.1),
                                 dpi=160, layout="constrained")
        for ax, (label, sim) in zip(axes, rows):
            ego_path = cf.LoggedPath(sim.log.actor_poses[hs.EGO_ID])
            hit_us = sim.result["hazard_hit_us"]
            shown = min(t, hit_us) if hit_us is not None else t
            s = centre if centre is not None else _hazard_s(sim, ego_path, shown)
            bev_panel(ax, sim, shown, View(ego_path, s), window, label + ("   IMPACT" if hit_us is not None and t >= hit_us else ""))
        fig.suptitle(note + "  Played at half speed.", fontsize=8, color="#555555")
        frames.append(_figure_rgb(fig))
        plt.close(fig)
    _write_video(frames, path, fps=FPS // 2)


def v2_occluded_pedestrian() -> None:
    case = HazardCase("d55c0611", "89b0de0c", "ped_walk_occluded")
    radar, fused = simulate(case, "radar"), simulate(case, "fused")
    onset = _hazard_onset_us(fused, case)
    _bev_video([("Radar only", radar), ("LiDAR + radar", fused)], onset - 3_000_000, fused.start_us + 2_000_000,
               ((-45, 12), (-9, 9)), _crossing_s(fused, case), OUT / "videos" / "V2_occluded_pedestrian.mp4",
               "Occluded pedestrian (injected, not rendered). Dashed: baseline ego. Lane lines from the logged ego path.")


def v3_cut_in() -> None:
    case = HazardCase("5c1b7555", "b7cf474a", "cut_in_5m")
    without_c, with_c = simulate(case, "fused", ["crossing_horizon_s=0"]), simulate(case, "fused")
    _bev_video([("Without crossing prediction", without_c), ("With crossing prediction", with_c)],
               with_c.start_us - 2_000_000, with_c.start_us + 4_000_000, ((-32, 18), (-7, 8)), None,
               OUT / "videos" / "V3_cut_in.mp4",
               "5 m cut-in at 33 m/s (injected, not rendered), view centred on the cut-in car. Dashed: baseline ego.")


def v4_real_crash(scene: str = "76c2bb81") -> None:
    """The logged at-fault crash, with the decisions the validator (LiDAR + radar)
    would have taken on it, replayed open loop."""
    run_dir = Path(glob.glob(f"benchmark/runs_core/alpamayo1_5__p*/rollouts/clipgt-{scene}*")[0]).parents[1]
    rollout = Rollout(run_dir.name, scene, run_dir.parent)
    times = rollout.step_times()
    crash = rollout.first_collision_step()
    steps = pl.read_csv("benchmark/runs_core/validator_replay_v13/steps_fused.csv", infer_schema_length=None).filter(
        pl.col("scene").str.contains(scene)).with_columns(pl.col("safe_speed_mps").cast(pl.Float64, strict=False))
    decisions = [(t, {"reason": r, "safe_speed": v, "violations": (viol or "").split("|") if viol else []})
                 for t, r, v, viol in steps.select("time_us", "reason", "safe_speed_mps", "violations").iter_rows()]
    # Warning time: how long the model's plan had been flagged without a break before impact.
    lead_s = rv._lead_time_s(steps.with_columns(pl.col("model_violations").fill_null("")), times[crash])
    first_warning = times[crash] - int(lead_s * 1e6) if lead_s > 0 else None
    capture = cv2.VideoCapture(str(rollout.video))
    start = max(crash - 100, 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for k in range(start, min(crash + 15, len(times))):
        ok, frame = capture.read()
        t = times[k]
        rgb = _label(cv2.cvtColor(frame[CAMERA_TILE], cv2.COLOR_BGR2RGB),
                     f"Logged baseline drive (at-fault crash)   t = {(t - times[0]) * 1e-6:4.1f} s", scale=0.6, origin=(12, 28))
        text, active = _decision_text(_decision_at(decisions, t))
        text = text.replace("Validator:", "Validator (replayed):")
        if first_warning is not None and t >= first_warning:
            text += f"   | first warning {(times[crash] - first_warning) * 1e-6:.1f} s before impact"
        if k >= crash:
            text = "IMPACT   " + text
        frames.append(np.vstack([rgb, _status_strip(rgb.shape[1], text, active)]))
    _write_video(frames, OUT / "videos" / "V4_real_crash_replayed.mp4")


def v5_normal_driving() -> None:
    """20 s of a crash-free closed-loop validator rollout: the one with the fewest
    interventions among those that had at least one."""
    run_dir = Path("benchmark/runs_core")
    candidates = []
    for d in glob.glob(str(run_dir / "alpamayo1_5_safety_fused__p1" / "rollouts" / "clipgt-*" / "*")):
        rollout = Rollout("alpamayo1_5_safety_fused__p1", Path(d).parent.name[7:15], run_dir, Path(d).name[:8])
        if rollout.first_collision_step() is not None:
            continue
        decisions = _logged_decisions(rollout)
        active = sum(d["reason"] == "speed_override" for _, d in decisions)
        if active:
            candidates.append((active / len(decisions), rollout, decisions))
    share, rollout, decisions = min(candidates, key=lambda c: c[0])
    times = rollout.step_times()
    capture = cv2.VideoCapture(str(rollout.video))
    frames = []
    for t in times[:200]:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = _label(cv2.cvtColor(frame[CAMERA_TILE], cv2.COLOR_BGR2RGB),
                     f"Closed loop with validator (LiDAR + radar)   t = {(t - times[0]) * 1e-6:4.1f} s", scale=0.6, origin=(12, 28))
        text, active = _decision_text(_decision_at(decisions, t))
        frames.append(np.vstack([rgb, _status_strip(rgb.shape[1], text, active)]))
    _write_video(frames, OUT / "videos" / "V5_normal_driving.mp4")
    print(f"V5: scene {rollout.scene}, validator intervened on {100 * share:.1f}% of steps")


# Videos V6/V7: the rendered camera beside a top-down panel in the ego frame
# (forward up), drawn from the rollout log: every candidate the policy sampled,
# its pick and the validator's, and (V7) LiDAR and radar.

PANEL_WINDOW = ((-13.0, 13.0), (-7.0, 38.0))  # (lateral, forward) m, drawn to scale
CANDIDATE_COLORS = {"all": "#adb5bd", "model": "#f4a261", "selected": "#2a9d8f"}


def _rig_from_local(drive: rv.DriveStep) -> np.ndarray:
    return np.linalg.inv(pose_from_grpc(drive.selected.poses[0].pose).as_se3().astype(np.float64))


def _path_rig(rig_from_local: np.ndarray, trajectory) -> np.ndarray:
    xyz = np.array([[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z, 1.0] for p in trajectory.poses])
    return (rig_from_local @ xyz.T).T[:, :2]


def _drive_at(drives: list[rv.DriveStep], time_us: int) -> int:
    """Index of the last Drive call at or before ``time_us`` (the first one before any)."""
    return max([i for i, d in enumerate(drives) if d.time_us <= time_us], default=0)


def _top_down(log: rv.RolloutLog, drive: rv.DriveStep, decision: dict, size_px: tuple[int, int],
              title: str, window: tuple = PANEL_WINDOW, lidar=None, radar=None, obstacles=()) -> np.ndarray:
    """The ego-frame panel at ``drive``: road users (true boxes), the sampled
    candidates, and optionally sensor points and sensed obstacles."""
    w, h = size_px
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.93))
    rig_from_local = _rig_from_local(drive)
    t = drive.time_us
    flip = lambda xy: np.column_stack([-xy[:, 1], xy[:, 0]])  # forward up, left to the left
    for actor_id, poses in log.actor_poses.items():
        if actor_id == hs.EGO_ID or not poses:
            continue
        key = min(poses, key=lambda k: abs(k - t))
        if abs(key - t) > 150_000:
            continue
        se3 = rig_from_local @ pose_from_grpc(poses[key]).as_se3().astype(np.float64)
        size = log.aabbs[actor_id]
        corners = cf._box_corners(se3[:2, 3], cf._yaw(se3), size.x, size.y)
        ax.add_patch(matplotlib.patches.Polygon(flip(corners), closed=True, facecolor="#e9ecef", edgecolor="#868e96", lw=0.8, zorder=2))
    ego = log.aabbs[hs.EGO_ID]
    ax.add_patch(matplotlib.patches.Polygon(flip(cf._box_corners(np.array([1.3, 0.0]), 0.0, ego.x, ego.y)), closed=True,
                                            facecolor="#457b9d", edgecolor="#1d3557", zorder=8))
    if lidar is not None and len(lidar):
        ax.scatter(-lidar[:, 1], lidar[:, 0], s=5, color="#212529", linewidths=0, zorder=3)
    if radar is not None and len(radar):
        ax.scatter(-radar[:, 1], radar[:, 0], s=22, marker="D", facecolor="none", edgecolor=SOURCE_COLORS["radar"], lw=1.2, zorder=4)
    for obstacle in obstacles:
        color = SOURCE_COLORS[obstacle["source"]]
        half_l, half_w = obstacle.get("half_length_m", 0.0), obstacle.get("half_width_m", 0.0)
        if half_l > 0.05 and half_w > 0.05:
            ax.add_patch(matplotlib.patches.Rectangle((-obstacle["y"] - half_w, obstacle["x"] - half_l), 2 * half_w, 2 * half_l,
                                                      fill=False, edgecolor=color, lw=1.6, zorder=5))
        else:
            ax.plot(-obstacle["y"], obstacle["x"], marker="x", color=color, ms=7, mew=1.8, zorder=5)
    paths = [flip(_path_rig(rig_from_local, c)) for c in drive.sampled]
    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color=CANDIDATE_COLORS["all"], lw=1.0, zorder=6)
    model, selected = decision["model_index"], decision["selected_index"]
    if paths:
        if model != selected:
            _plot_candidate(ax, drive.sampled[model], paths[model], CANDIDATE_COLORS["model"], (0, (4, 2)))
        _plot_candidate(ax, drive.sampled[selected], paths[selected], CANDIDATE_COLORS["selected"], "-")
    (x0, x1), (y0, y1) = window
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#ced4da")
    fig.text(0.03, 0.965, title, fontsize=8, va="center")
    rgb = _figure_rgb(fig)
    plt.close(fig)
    return cv2.resize(rgb, size_px, interpolation=cv2.INTER_AREA)


def _plot_candidate(ax, trajectory, path: np.ndarray, color: str, style) -> None:
    """``path`` with a tick every second, so that slower candidates show closer ticks."""
    ax.plot(path[:, 0], path[:, 1], color=color, lw=2.5, ls=style, zorder=7)
    t0 = trajectory.poses[0].timestamp_us
    dt = np.array([p.timestamp_us - t0 for p in trajectory.poses])
    ticks = [int(np.argmin(np.abs(dt - s * 1_000_000))) for s in range(1, int(dt[-1] // 1_000_000) + 1)]
    ax.scatter(path[ticks, 0], path[ticks, 1], s=30, color=color, edgecolor="white", lw=0.8, zorder=7.5)


def _candidate_speed(trajectory) -> float:
    """Mean speed over the candidate's first second, m/s."""
    poses = trajectory.poses
    t0 = poses[0].timestamp_us
    end = max(i for i, p in enumerate(poses) if p.timestamp_us - t0 <= 1_000_000)
    a, b = poses[0].pose.vec, poses[end].pose.vec
    return math.hypot(b.x - a.x, b.y - a.y) / ((poses[end].timestamp_us - t0) * 1e-6)


def _legend_strip(width: int, items: list[tuple[str, tuple[int, int, int], str]]) -> np.ndarray:
    """A light strip of legend entries, wrapped onto as many rows as needed:
    (label, RGB colour, 'line' | 'dash' | 'box' | 'fill' | 'x' | 'dot' | 'diamond')."""
    font, scale, row_h = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 28
    widths = [34 + cv2.getTextSize(label, font, scale, 1)[0][0] + 20 for label, _, _ in items]
    rows, x = 1, 14
    for w in widths:
        if x + w > width:
            rows, x = rows + 1, 14
        x += w
    strip = np.full((rows * row_h + 6, width, 3), 248, dtype=np.uint8)
    x, y = 14, 3 + row_h // 2
    for (label, color, kind), w in zip(items, widths):
        if x + w > width:
            x, y = 14, y + row_h
        if kind in ("line", "dash"):
            for x0 in range(x, x + 28, 8 if kind == "dash" else 28):
                cv2.line(strip, (x0, y), (min(x0 + (5 if kind == "dash" else 28), x + 28), y), color, 3, cv2.LINE_AA)
        elif kind == "box":
            cv2.rectangle(strip, (x + 6, y - 7), (x + 22, y + 7), color, 2)
        elif kind == "fill":
            cv2.rectangle(strip, (x + 6, y - 7), (x + 22, y + 7), color, -1)
        elif kind == "x":
            cv2.line(strip, (x + 8, y - 6), (x + 20, y + 6), color, 2, cv2.LINE_AA)
            cv2.line(strip, (x + 8, y + 6), (x + 20, y - 6), color, 2, cv2.LINE_AA)
        elif kind == "dot":
            cv2.circle(strip, (x + 14, y), 3, color, -1, cv2.LINE_AA)
        elif kind == "diamond":
            pts = np.array([[x + 14, y - 7], [x + 21, y], [x + 14, y + 7], [x + 7, y]])
            cv2.polylines(strip, [pts], True, color, 2, cv2.LINE_AA)
        cv2.putText(strip, label, (x + 34, y + 5), font, scale, (52, 58, 64), 1, cv2.LINE_AA)
        x += w
    return strip


def _hex_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _switch_text(decision: dict, drive: rv.DriveStep) -> tuple[str, bool]:
    text, active = _decision_text(decision)
    model, selected = decision["model_index"], decision["selected_index"]
    if selected != model:
        text = (f"Validator: policy's candidate ({_candidate_speed(drive.sampled[model]):.1f} m/s) flagged{_constraints(decision['model_violations'])}; "
                f"switched to its safe candidate ({_candidate_speed(drive.sampled[selected]):.1f} m/s)")
        active = True
    return text, active


def _camera_and_panel(rollout: Rollout, steps: range, title: str, path: Path, sensors: bool, fps: int,
                      window: tuple = PANEL_WINDOW, hold_switches: bool = False) -> None:
    """Frames: [labelled camera | top-down panel], legend strip, status strip; with
    ``hold_switches``, each step where the validator switches candidate is held 1 s."""
    times = rollout.step_times()
    decisions = _logged_decisions(rollout)
    log = asyncio.run(rv._read_log(rollout.dir / "rollout.asl"))
    assert len(decisions) == len(log.drives), "one validator record per Drive call"
    crash = rollout.first_collision_step()
    sensed = {}
    if sensors:
        from alpasim_driver.safety_filter import SafetyFilter

        safety_filter = SafetyFilter(rv.validator_config([]), rv.OUTPUT_FREQUENCY_HZ)
        traffic, ego_trajectory = rv._traffic(log), rv._trajectory(log.ego_poses)
        radars = rv._load_radars(rv.DEFAULT_RADAR_CONFIG)
        last = max(times[k] for k in steps)
        for i, drive in enumerate(log.drives):
            t = drive.time_us
            if t > last or t not in log.ego_speed:
                continue
            rig_from_local = _rig_from_local(drive)
            obstacles = rv.sense(safety_filter, "fused", traffic, ego_trajectory, t, rig_from_local, log.ego_speed[t],
                                 log.ego_yaw_rate[t], radars, seed=0, mode=rv.FIXED_MODES["fused"]).obstacles
            if t >= times[steps[0]] - 200_000:
                sensed[i] = (_lidar_points(log, t, rig_from_local, None), _radar_points(log, t, None), obstacles)
    capture = cv2.VideoCapture(str(rollout.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, steps[0])
    frames = []
    for k in steps:
        ok, frame = capture.read()
        if not ok:
            break
        t = times[k]
        impact = "   IMPACT" if crash is not None and k >= crash else ""
        camera = _label(cv2.cvtColor(frame[CAMERA_TILE], cv2.COLOR_BGR2RGB),
                        f"{title}   t = {(t - times[0]) * 1e-6:4.1f} s{impact}", scale=0.6, origin=(12, 28))
        i = _drive_at(log.drives, t)
        decision = decisions[i][1]
        (x0, x1), (y0, y1) = window
        size = (int(camera.shape[0] * 0.93 * (x1 - x0) / (y1 - y0)), camera.shape[0])
        if sensors:
            lidar, radar, obstacles = sensed.get(i, (None, None, ()))
            panel = _top_down(log, log.drives[i], decision, size, "LiDAR + radar (re-simulated)",
                              window, lidar=lidar, radar=radar, obstacles=obstacles)
        else:
            panel = _top_down(log, log.drives[i], decision, size, "Policy's 6 candidates", window)
        row = np.hstack([camera, np.full((camera.shape[0], 6, 3), 255, np.uint8), panel])
        text, active = _switch_text(decision, log.drives[i])
        switched = decision["selected_index"] != decision["model_index"]
        frames += [np.vstack([row, _video_legend(row.shape[1], sensors), _status_strip(row.shape[1], text, active)])] * (
            fps if hold_switches and switched else 1)
    frames += [frames[-1]] * int(1.5 * fps)
    _write_video(frames, path, fps=fps)


def _video_legend(width: int, sensors: bool) -> np.ndarray:
    items = [("ego", _hex_rgb("#457b9d"), "fill"), ("road users (true)", _hex_rgb("#868e96"), "box"),
             ("candidates", _hex_rgb(CANDIDATE_COLORS["all"]), "line"),
             ("policy's pick (if rejected)", _hex_rgb(CANDIDATE_COLORS["model"]), "dash"),
             ("driven", _hex_rgb(CANDIDATE_COLORS["selected"]), "line")]
    if sensors:
        items = items[:2] + [("LiDAR returns", (73, 80, 87), "dot"), ("radar detections", _hex_rgb(SOURCE_COLORS["radar"]), "diamond"),
                             ("track: LiDAR", _hex_rgb(SOURCE_COLORS["lidar"]), "box"),
                             ("track: LiDAR+radar", _hex_rgb(SOURCE_COLORS["lidar+radar"]), "box"),
                             ("track: radar only", _hex_rgb(SOURCE_COLORS["radar"]), "x"),
                             ("driven path", _hex_rgb(CANDIDATE_COLORS["selected"]), "line")]
    return _legend_strip(width, items)


def v6_trajectory_switch(run: str = "alpamayo1_5_safety_fused__p1", scene: str = "05f35348", rollout: str = "0e56fe48",
                         first: int = 150, last: int = 182) -> None:
    """A closed-loop highway rollout behind a lead vehicle: the validator brakes,
    then rejects the policy's pick and drives a candidate that keeps clear of the
    lead vehicle; half speed, each switch held 1 s."""
    r = Rollout(run, scene, Path("benchmark/runs_core"), rollout)
    _camera_and_panel(r, range(first, last + 1), "Closed loop with validator", OUT / "videos" / "V6_trajectory_switch.mp4",
                      sensors=False, fps=FPS // 2, window=((-11.0, 11.0), (-6.0, 50.0)), hold_switches=True)


def v7_lidar_radar(scene: str = "6250146a") -> None:
    """The braking-car rollout of V1 (with validator) beside the LiDAR and radar."""
    r = Rollout("alpamayo1_5_safety_fused_leadbrake__s2", scene)
    _camera_and_panel(r, range(0, V1_LAST_STEP + 1), "With validator (LiDAR + radar)",
                      OUT / "videos" / "V7_lidar_radar.mp4", sensors=True, fps=FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    items = {
        "teaser": teaser,
        "b2": b2_occluded_pedestrian,
        "b3": b3_cut_in,
        "b4": b4_braking_limited_miss,
        "b5": b5_speed_vs_time,
        "b8": b8_rain_scene,
        "v1": v1_braking_car,
        "v2": v2_occluded_pedestrian,
        "v3": v3_cut_in,
        "v4": v4_real_crash,
        "v5": v5_normal_driving,
        "v6": v6_trajectory_switch,
        "v7": v7_lidar_radar,
    }
    parser.add_argument("items", nargs="+", choices=list(items))
    for item in parser.parse_args().items:
        items[item]()


if __name__ == "__main__":
    main()
