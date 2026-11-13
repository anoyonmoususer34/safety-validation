"""Replay finished rollouts through the safety filter, without a GPU.

Each rollout.asl holds the model's sampled candidates at every Drive call, the
ego state and every actor's pose.  This feeds them to the driver's real
SafetyFilter with obstacles from one of these sensors:
  gt      every actor at its true centre and velocity, confidence 1
  radar   the runtime's simulated radar with the safety_validator_radar settings
  lidar   the driver's real LiDAR detector on points ray-cast against the actor
          boxes, in place of the renderer's spin
  fused   both of the above
  policy  whichever of them exp/sim/safety_validator_weather.yaml picks for the
          weather label, through the driver's own label and policy code

The policy labels are the prompt's answers from benchmark/weather_label_eval.py
(--weather-labels), which needs a GPU once.  Without that file, the hand labels of
--scene-weather stand in, as a prompt that is always right.  --scene-weather also
degrades both sensors to each scene's actual weather, as the runtime does.

Open loop: the logged car does not react to the filter, so this shows how often
and when the filter would act on these logs, not whether it prevents a crash.
Obstacles differ from a closed-loop run's after the first intervention.  The
ray-cast LiDAR sees only the logged actors, not walls, kerbs or vegetation, so it
has fewer false detections than the renderer's spin would.

Writes RUNS_DIR/OUTPUT_NAME/{steps,rollouts}_SENSOR.csv and prints a table.
Run on a LOGIN node, from the repo root:
    uv run python benchmark/replay_validator.py benchmark/runs_core --arm alpamayo1_5
    uv run python benchmark/replay_validator.py benchmark/runs \\
        --sensors lidar radar policy --scene-weather src/wizard/configs/exp/weather/dev_scenes.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import asyncio
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# Each worker is single-threaded; login nodes cap the threads per user.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Polars, the Rust log reader and their pools otherwise start a thread per core
# in every worker; login nodes cap the threads per user.
for _pool in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
    os.environ.setdefault(_pool, "1")

import numpy as np
import polars as pl
import yaml
from alpasim_driver.models.base import ModelPrediction
from alpasim_driver.safety_filter import SafetyFilter, SensedObstacles
from alpasim_driver.scene_labels import LabelEstimate, SensorMode, parse_sensor_policy
from alpasim_driver.schema import SafetyValidatorConfig, SceneLabelConfig
from alpasim_grpc.v0 import common_pb2
from alpasim_runtime.config import PoseConfig, RadarConfig
from alpasim_runtime.radar import simulate_radar_scan
from alpasim_runtime.sensor_weather import (
    WEATHER_EFFECTS,
    degrade_lidar_payload,
    degrade_radar,
)
from alpasim_utils.geometry import Pose, Trajectory, pose_from_grpc, trajectory_from_grpc
from alpasim_utils.lidar_payload import pack_lidar_spin
from alpasim_utils.logs import async_read_pb_log
from alpasim_utils.radar_payload import pack_radar_scan
from alpasim_utils.sensor_payload import pack_sensor_payloads
from alpasim_utils.scenario import AABB, TrafficObject, TrafficObjects

EGO_ID = "EGO"
OUTPUT_FREQUENCY_HZ = 10  # Alpamayo waypoint rate
VELOCITY_DT_US = 50_000  # same backward difference as alpasim_runtime.radar
COLLISION_METRICS = ("collision_front", "collision_lateral", "collision_rear")
DEFAULT_RADAR_CONFIG = Path("benchmark/radar_config.yaml")
POLICY_CONFIG = Path("src/wizard/configs/exp/sim/safety_validator_weather.yaml")
SENSORS = ("gt", "radar", "lidar", "fused", "policy")

# Roof LiDAR as calibrated on the Hyperion rigs, with a 64-beam scan pattern.
LIDAR_IN_RIG = np.array([1.17, 0.0, 1.93])
RIG_TO_LIDAR = Pose(
    LIDAR_IN_RIG.astype(np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
)
LIDAR_ELEVATIONS_RAD = np.radians(np.linspace(-25.0, 15.0, 64))
LIDAR_AZIMUTHS_RAD = np.radians(np.arange(-180.0, 180.0, 0.2))
LIDAR_MAX_RANGE_M = 80.0
LIDAR_RANGE_NOISE_M = 0.02


@dataclass
class DriveStep:
    time_us: int
    selected: common_pb2.Trajectory
    sampled: list[common_pb2.Trajectory]


@dataclass
class RolloutLog:
    aabbs: dict[str, AABB] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    actor_poses: dict[str, dict[int, common_pb2.Pose]] = field(default_factory=dict)
    ego_poses: dict[int, common_pb2.Pose] = field(default_factory=dict)
    ego_speed: dict[int, float] = field(default_factory=dict)
    ego_yaw_rate: dict[int, float] = field(default_factory=dict)
    drives: list[DriveStep] = field(default_factory=list)


async def _read_log(path: Path) -> RolloutLog:
    log = RolloutLog()
    pending_time_us = None
    async for entry in async_read_pb_log(str(path)):
        kind = entry.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            for actor in entry.rollout_metadata.actor_definitions.actor_aabb:
                size = actor.aabb
                log.aabbs[actor.actor_id] = AABB(size.size_x, size.size_y, size.size_z)
                log.labels[actor.actor_id] = actor.actor_label
        elif kind == "actor_poses":
            time_us = entry.actor_poses.timestamp_us
            for actor in entry.actor_poses.actor_poses:
                log.actor_poses.setdefault(actor.actor_id, {})[time_us] = actor.actor_pose
        elif kind == "driver_ego_trajectory":
            msg = entry.driver_ego_trajectory
            last = msg.trajectory.poses[-1]
            state = msg.dynamic_states[-1]
            velocity = state.linear_velocity
            log.ego_poses[last.timestamp_us] = last.pose
            log.ego_speed[last.timestamp_us] = float(np.hypot(velocity.x, velocity.y))
            log.ego_yaw_rate[last.timestamp_us] = float(state.angular_velocity.z)
        elif kind == "driver_request":
            pending_time_us = entry.driver_request.time_now_us
        elif kind == "driver_return":
            response = entry.driver_return
            log.drives.append(
                DriveStep(
                    pending_time_us,
                    response.trajectory,
                    list(response.debug_info.sampled_trajectories),
                )
            )
    return log


def _trajectory(poses: dict[int, common_pb2.Pose]) -> Trajectory:
    return trajectory_from_grpc(
        common_pb2.Trajectory(
            poses=[
                common_pb2.PoseAtTime(pose=poses[t], timestamp_us=t)
                for t in sorted(poses)
            ]
        )
    )


def _traffic(log: RolloutLog) -> TrafficObjects:
    return TrafficObjects(
        **{
            actor_id: TrafficObject(
                track_id=actor_id,
                aabb=log.aabbs[actor_id],
                trajectory=_trajectory(poses),
                is_static=False,
                label_class=log.labels[actor_id],
            )
            for actor_id, poses in log.actor_poses.items()
            if actor_id != EGO_ID and len(poses) >= 2
        }
    )


def _waypoints_xyz(trajectory: common_pb2.Trajectory) -> np.ndarray:
    return np.array(
        [[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in trajectory.poses]
    )


def _prediction(
    drive: DriveStep, rig_from_local: np.ndarray, single_candidate: bool = False
) -> ModelPrediction:
    """The Drive call's candidates back in the rig frame of the model's t0; with
    ``single_candidate``, only the model's own, as a one-sample policy would give."""
    # A logged trajectory leads with the rig pose at t0, then the model's waypoints.
    selected = _waypoints_xyz(drive.selected)[1:]
    local = np.stack([_waypoints_xyz(t)[1:] for t in drive.sampled])
    # The driven plan may be a smoothed copy of its candidate, so match by distance.
    selected_index = int(np.argmin(np.linalg.norm(local - selected, axis=2).mean(axis=1)))
    homogeneous = np.concatenate([local, np.ones((*local.shape[:2], 1))], axis=2)
    positions = (homogeneous @ rig_from_local.T)[..., :3]
    if single_candidate:
        positions = positions[selected_index : selected_index + 1]
        selected_index = 0
    rotations = np.tile(np.eye(3), (*positions.shape[:2], 1, 1))
    return ModelPrediction(
        candidate_positions=positions,
        candidate_rotations=rotations,
        selected_index=selected_index,
    )


def _gt_obstacles(
    traffic: TrafficObjects, time_us: int, rig_from_local: np.ndarray
) -> list[dict]:
    obstacles = []
    for actor_id, obj in traffic.items():
        trajectory = obj.trajectory
        if time_us not in trajectory.time_range_us:
            continue
        position = trajectory.interpolate_pose(time_us).as_se3()[:, 3].astype(np.float64)
        prev_us = max(trajectory.time_range_us.start, time_us - VELOCITY_DT_US)
        velocity = np.zeros(2)
        if prev_us < time_us:
            prev = trajectory.interpolate_pose(prev_us).as_se3()[:2, 3]
            velocity = (position[:2] - prev) / ((time_us - prev_us) * 1e-6)
        x, y = (rig_from_local @ position)[:2]
        vx, vy = rig_from_local[:2, :2] @ velocity
        obstacles.append(
            {
                "id": actor_id,
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy),
                "half_length_m": obj.aabb.x / 2,
                "half_width_m": obj.aabb.y / 2,
                "confidence": 1.0,
                "source": "gt",
            }
        )
    return obstacles


def _lidar_spin_points(
    traffic: TrafficObjects,
    time_us: int,
    rig_from_local: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """(N, 3) sensor-frame points: rays of the scan pattern against the actor boxes."""
    elevations, azimuths = LIDAR_ELEVATIONS_RAD, LIDAR_AZIMUTHS_RAD
    azimuth_step = azimuths[1] - azimuths[0]
    ranges = np.full((len(elevations), len(azimuths)), np.inf)
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    for obj in traffic.values():
        if time_us not in obj.trajectory.time_range_us:
            continue
        sensor_from_actor = rig_from_local @ obj.trajectory.interpolate_pose(
            time_us
        ).as_se3().astype(np.float64)
        sensor_from_actor[:3, 3] -= LIDAR_IN_RIG
        half = np.array([obj.aabb.x, obj.aabb.y, obj.aabb.z]) / 2
        corners = signs * half @ sensor_from_actor[:3, :3].T + sensor_from_actor[:3, 3]
        horizontal = np.hypot(corners[:, 0], corners[:, 1])
        if horizontal.min() > LIDAR_MAX_RANGE_M:
            continue

        # The rays that can reach the box, by azimuth around its centre (which
        # handles boxes straddling +-180 deg) and by elevation.
        center = sensor_from_actor[:3, 3]
        center_azimuth = np.arctan2(center[1], center[0])
        offsets = np.arctan2(corners[:, 1], corners[:, 0]) - center_azimuth
        offsets = (offsets + np.pi) % (2 * np.pi) - np.pi
        first = int(np.floor((center_azimuth + offsets.min() + np.pi) / azimuth_step))
        last = int(np.ceil((center_azimuth + offsets.max() + np.pi) / azimuth_step))
        azimuth_idx = np.arange(first, last + 1) % len(azimuths)
        corner_elevations = np.arctan2(corners[:, 2], horizontal)
        elevation_idx = np.flatnonzero(
            (elevations >= corner_elevations.min())
            & (elevations <= corner_elevations.max())
        )
        if len(elevation_idx) == 0:
            continue

        el, az = np.meshgrid(elevations[elevation_idx], azimuths[azimuth_idx], indexing="ij")
        rays = np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
        )
        # Slab test in the actor's frame.
        rotation = sensor_from_actor[:3, :3]
        origin = -rotation.T @ center
        directions = rays @ rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            near = (-half - origin) / directions
            far = (half - origin) / directions
        t_near = np.nanmax(np.minimum(near, far), axis=-1)
        t_far = np.nanmin(np.maximum(near, far), axis=-1)
        hit = (t_near <= t_far) & (t_near > 0) & (t_near <= LIDAR_MAX_RANGE_M)
        block = ranges[np.ix_(elevation_idx, azimuth_idx)]
        ranges[np.ix_(elevation_idx, azimuth_idx)] = np.minimum(
            block, np.where(hit, t_near, np.inf)
        )

    el_i, az_i = np.nonzero(np.isfinite(ranges))
    el, az = elevations[el_i], azimuths[az_i]
    measured = ranges[el_i, az_i] + rng.normal(0.0, LIDAR_RANGE_NOISE_M, len(el_i))
    return (
        measured[:, None]
        * np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    ).astype(np.float32)


class PolicyLabels:
    """The weather label over a rollout, as the driver would hold it: the prompt's
    timed answers through LabelEstimate, or a fixed label standing in for them."""

    def __init__(self, answers: list[dict] | None, fixed_label: str | None) -> None:
        self._answers = sorted(answers or [], key=lambda a: a["time_us"])
        self._fixed_label = fixed_label
        self._estimate = LabelEstimate(SceneLabelConfig(enabled=True))

    def label(self, time_us: int) -> str:
        if self._fixed_label is not None:
            return self._fixed_label
        while self._answers and self._answers[0]["time_us"] <= time_us:
            answer = self._answers.pop(0)
            self._estimate.update(answer["time_us"], answer["probabilities"])
        return self._estimate.label


def _closest_ahead(debug: dict) -> dict | None:
    """The closest obstacle the filter judged in the corridor ahead, from its
    validator trace or, for the RSS baseline, its RSS assessments."""
    if "rss_assessments" in debug:
        relevant = [
            {"id": a["id"], "longitudinal_distance": a["distance"]} for a in debug["rss_assessments"]
        ]
    else:
        relevant = debug["validator_result"]["trace_data"]["relevant_obstacles"]
    return min(relevant, key=lambda o: o["longitudinal_distance"], default=None)


def sense(
    safety_filter: SafetyFilter,
    sensor: str,
    traffic: TrafficObjects,
    ego_trajectory: Trajectory,
    time_us: int,
    rig_from_local: np.ndarray,
    speed: float,
    yaw_rate: float,
    radars: list[RadarConfig],
    seed: int,
    effects=None,
    mode: SensorMode | None = None,
    lidar_dropout: float = 0.0,
) -> SensedObstacles:
    """What ``safety_filter`` senses of ``traffic`` from the rig at ``rig_from_local``:
    ``sensor`` gt, or the simulated LiDAR and radar that ``mode`` selects.  The radar
    takes the ego's velocity from ``ego_trajectory``; ``lidar_dropout`` is the
    fraction of LiDAR returns lost at random."""
    if sensor == "gt":
        return SensedObstacles(
            SensorMode.LIDAR, _gt_obstacles(traffic, time_us, rig_from_local), {}
        )
    payloads = []
    if mode in (SensorMode.LIDAR, SensorMode.FUSED):
        points = _lidar_spin_points(
            traffic, time_us, rig_from_local, np.random.default_rng([seed, time_us, 1])
        )
        if lidar_dropout > 0.0:
            keep = np.random.default_rng([seed, time_us, 4]).random(len(points)) >= lidar_dropout
            points = points[keep]
        payload = pack_lidar_spin(points.tobytes(), len(points), time_us, RIG_TO_LIDAR)
        if effects is not None:
            payload = degrade_lidar_payload(
                payload, effects, np.random.default_rng([seed, time_us, 2])
            )
        payloads.append(payload)
    if mode in (SensorMode.RADAR, SensorMode.FUSED):
        scan = simulate_radar_scan(
            radars, ego_trajectory, traffic, time_us, np.random.default_rng([seed, time_us])
        )
        if effects is not None:
            scan = degrade_radar(scan, effects, np.random.default_rng([seed, time_us, 3]))
        payloads.append(pack_radar_scan(scan))
    return safety_filter.sense(pack_sensor_payloads(payloads), speed, yaw_rate, mode)


FIXED_MODES = {"radar": SensorMode.RADAR, "lidar": SensorMode.LIDAR, "fused": SensorMode.FUSED}


def replay_steps(
    log: RolloutLog,
    sensor: str,
    radars: list[RadarConfig],
    cfg: SafetyValidatorConfig,
    seed: int,
    scene_weather: str | None = None,
    policy: dict[str, SensorMode] | None = None,
    labels: PolicyLabels | None = None,
    single_candidate: bool = False,
) -> list[dict]:
    safety_filter = SafetyFilter(cfg, OUTPUT_FREQUENCY_HZ)
    traffic = _traffic(log)
    ego_trajectory = _trajectory(log.ego_poses)
    effects = WEATHER_EFFECTS[scene_weather] if scene_weather else None

    steps = []
    for drive in log.drives:
        t0 = drive.selected.poses[0]
        time_us = t0.timestamp_us
        if time_us not in log.ego_speed:
            continue  # before the first logged ego state
        speed = log.ego_speed[time_us]
        rig_from_local = np.linalg.inv(pose_from_grpc(t0.pose).as_se3().astype(np.float64))
        label = None
        mode = None
        if sensor == "policy":
            label = labels.label(time_us)
            mode = policy[label]
        elif sensor != "gt":
            mode = FIXED_MODES[sensor]
        sensed = sense(
            safety_filter, sensor, traffic, ego_trajectory, time_us, rig_from_local,
            speed, log.ego_yaw_rate[time_us], radars, seed, effects, mode,
        )
        prediction = _prediction(drive, rig_from_local, single_candidate)
        model_index = prediction.selected_index
        driven, debug = safety_filter.apply(prediction, sensed, speed)

        closest = _closest_ahead(debug)
        steps.append(
            {
                "time_us": time_us,
                "label": label,
                "sensor_mode": str(sensed.mode),
                "speed_mps": speed,
                "reason": debug["reason"],
                "switched": driven.selected_index != model_index,
                "safe_speed_mps": debug.get("safe_speed"),
                "model_violations": "|".join(debug["model_violations"]),
                "violations": "|".join(debug["violations"]),
                "num_obstacles": len(sensed.obstacles),
                "closest_gap_m": closest["longitudinal_distance"] if closest else None,
                "closest_id": closest["id"] if closest else None,
            }
        )
    return steps


def _first_collision(metrics_path: Path) -> tuple[int | None, str | None]:
    """Time and type of the rollout's first collision, per the eval metrics."""
    hits = (
        pl.read_parquet(metrics_path)
        .filter(pl.col("name").is_in(COLLISION_METRICS) & (pl.col("values") > 0))
        .sort("timestamps_us")
    )
    if hits.is_empty():
        return None, None
    return int(hits["timestamps_us"][0]), hits["name"][0].removeprefix("collision_")


def _lead_time_s(steps: pl.DataFrame, collision_us: int) -> float:
    """How long the model's plan had been flagged without a break before the crash."""
    before = steps.filter(pl.col("time_us") < collision_us).sort("time_us", descending=True)
    start_us = None
    for time_us, flagged in zip(before["time_us"], before["model_violations"] != ""):
        if not flagged:
            break
        start_us = time_us
    return 0.0 if start_us is None else (collision_us - start_us) * 1e-6


def summarize_rollout(
    steps: pl.DataFrame, scene: str, rollout: str, run: str, at_fault: bool, metrics_path: Path
) -> dict:
    flagged = steps["model_violations"] != ""
    collision_us, collision_type = _first_collision(metrics_path)
    row = {
        "scene": scene,
        "rollout": rollout,
        "run": run,
        "steps": steps.height,
        "at_fault": at_fault,
        "collision": collision_type,
        "plan_flagged_pct": 100.0 * flagged.mean(),
        "switched_pct": 100.0 * steps["switched"].mean(),
        "speed_override_pct": 100.0 * (steps["reason"] == "speed_override").mean(),
        # Overrides below the ego's current speed; the others only cap a plan
        # that would speed up.
        "braking_override_pct": 100.0
        * (
            (steps["reason"] == "speed_override")
            & (steps["safe_speed_mps"].cast(pl.Float64, strict=False) < steps["speed_mps"])
        ).mean(),
        "lead_time_s": None,
        "last_speed_mps": None,
        "last_closest_gap_m": None,
    }
    for name in ("EMERGENCY_TTC", "STOPPING_DISTANCE", "LOW_TTC", "REAR_COLLISION"):
        row[f"{name.lower()}_pct"] = (
            100.0 * steps["model_violations"].str.contains(name).mean()
        )
    if collision_us is not None:
        before = steps.filter(pl.col("time_us") < collision_us)
        row["lead_time_s"] = _lead_time_s(steps, collision_us)
        if not before.is_empty():
            last = before.row(-1, named=True)
            row["last_speed_mps"] = last["speed_mps"]
            row["last_closest_gap_m"] = last["closest_gap_m"]
    return row


def _replay_rollout(job: dict) -> tuple[list[dict], dict]:
    asl_path = job["asl_path"]
    scene, rollout = asl_path.parents[1].name, asl_path.parent.name
    log = asyncio.run(_read_log(asl_path))
    scene_weather = job["scene_weather"].get(scene)
    labels = None
    if job["sensor"] == "policy":
        answers = job["weather_answers"].get(scene)
        # Without the prompt's answers, a prompt that is always right.
        labels = PolicyLabels(answers, None if answers else scene_weather or "clear")
    steps = replay_steps(
        log,
        job["sensor"],
        job["radars"],
        job["cfg"],
        job["seed"],
        scene_weather,
        job["policy"],
        labels,
        job["single_candidate"],
    )
    frame = pl.DataFrame(steps, infer_schema_length=None)
    summary = summarize_rollout(
        frame, scene, rollout, job["run"], job["at_fault"], asl_path.parent / "metrics.parquet"
    )
    return [{"scene": scene, "rollout": rollout, **step} for step in steps], summary


def validator_config(settings: list[str]) -> SafetyValidatorConfig:
    """The enabled validator with ``key=value`` settings (field names of
    SafetyValidatorConfig) in place of its defaults."""
    types = {f.name: f.type for f in dataclasses.fields(SafetyValidatorConfig)}
    values = {}
    for setting in settings:
        key, value = setting.split("=", 1)
        kind = types[key]
        values[key] = value.lower() == "true" if kind in (bool, "bool") else (
            float(value) if kind in (float, "float") else int(value) if kind in (int, "int") else value
        )
    return SafetyValidatorConfig(enabled=True, **values)


def _load_radars(path: Path) -> list[RadarConfig]:
    config = yaml.safe_load(path.read_text())
    return [
        RadarConfig(**{**radar, "rig_to_radar": PoseConfig(**radar["rig_to_radar"])})
        for radar in config["simulation_config"]["radars"]
    ]


def _load_policy() -> dict[str, SensorMode]:
    config = yaml.safe_load(POLICY_CONFIG.read_text())
    return parse_sensor_policy(config["driver"]["safety_validator"]["weather_sensor_policy"])


def _jobs(runs_dir: Path, arm: str, sensor: str, shared: dict) -> list[dict]:
    jobs = []
    run_dirs = sorted(
        d for d in runs_dir.glob(f"{arm}*")
        if d.name == arm or d.name.startswith(f"{arm}__p")
    )
    for run_dir in run_dirs:
        summary_path = run_dir / "aggregate" / "results-summary.json"
        if not summary_path.exists():
            print(f"Skipping {run_dir.name}: no aggregate results yet")
            continue
        rollouts = json.loads(summary_path.read_text())["rollouts"]
        at_fault = {r["rollout_id"]: r["metrics"]["collision_at_fault"] > 0 for r in rollouts}
        for asl_path in sorted(run_dir.glob("rollouts/*/*/rollout.asl")):
            jobs.append(
                {
                    "asl_path": asl_path,
                    "run": run_dir.name,
                    "sensor": sensor,
                    "at_fault": at_fault[asl_path.parent.name],
                    **shared,
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--arm", default="alpamayo1_5", help="arm name, without __pK")
    parser.add_argument("--sensors", nargs="+", default=list(SENSORS), choices=SENSORS)
    parser.add_argument("--scene-weather", type=Path, help="exp/weather preset")
    parser.add_argument("--weather-labels", type=Path, help="weather_label_eval.py output")
    parser.add_argument("--radar-config", type=Path, default=DEFAULT_RADAR_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="sensor noise seed")
    parser.add_argument("--limit", type=int, help="replay only the first N rollouts")
    parser.add_argument("--output-name", default="validator_replay", help="directory under RUNS_DIR")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="a SafetyValidatorConfig setting in place of its default, e.g. max_decel=8",
    )
    parser.add_argument(
        "--single-candidate", action="store_true",
        help="give the filter only the policy's own plan, as a one-sample policy would",
    )
    args = parser.parse_args()

    scene_weather = {}
    if args.scene_weather:
        scene_weather = yaml.safe_load(args.scene_weather.read_text())["runtime"][
            "simulation_config"
        ]["scene_weather"]
    weather_answers = {}
    if args.weather_labels:
        weather_answers = {
            scene["scene_id"]: scene["answers"]
            for scene in json.loads(args.weather_labels.read_text())["scenes"]
        }
    shared = {
        "radars": _load_radars(args.radar_config),
        "seed": args.seed,
        "scene_weather": scene_weather,
        "weather_answers": weather_answers,
        "policy": _load_policy(),
        "cfg": validator_config(args.set),
        "single_candidate": args.single_candidate,
    }
    output_dir = args.runs_dir / args.output_name
    output_dir.mkdir(exist_ok=True)
    comparison = []
    # Spawned, not forked: forking after polars has started its threads deadlocks.
    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    for sensor in args.sensors:
        jobs = _jobs(args.runs_dir, args.arm, sensor, shared)[: args.limit]
        print(f"== {sensor}: replaying {len(jobs)} rollouts", flush=True)
        results = list(pool.map(_replay_rollout, jobs))
        steps = pl.DataFrame(
            [step for rollout_steps, _ in results for step in rollout_steps],
            infer_schema_length=None,
        )
        rollouts = pl.DataFrame([summary for _, summary in results], infer_schema_length=None)
        steps.write_csv(output_dir / f"steps_{sensor}.csv")
        rollouts.write_csv(output_dir / f"rollouts_{sensor}.csv")
        with pl.Config(
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            float_precision=1,
            tbl_cols=-1,
            tbl_rows=-1,
            tbl_width_chars=200,
            fmt_str_lengths=20,
        ):
            print(
                rollouts.with_columns(pl.col("scene").str.slice(7, 8)).select(
                    "scene", "at_fault", "collision", "plan_flagged_pct", "switched_pct",
                    "speed_override_pct", "lead_time_s", "last_speed_mps",
                    "last_closest_gap_m",
                )
            )
        print(f"Wrote {output_dir}/steps_{sensor}.csv and rollouts_{sensor}.csv")
        clean = rollouts.filter(pl.col("collision").is_null())
        comparison.append(
            {
                "sensor": sensor,
                "flagged_pct": 100.0 * (steps["model_violations"] != "").mean(),
                "switched_pct": 100.0 * steps["switched"].mean(),
                "clean_rollouts_over_25pct": int((clean["plan_flagged_pct"] >= 25).sum()),
                "clean_rollouts": clean.height,
                "at_fault_lead_s": rollouts.filter(pl.col("at_fault"))["lead_time_s"].to_list(),
                "modes": dict(steps["sensor_mode"].value_counts().rows()),
            }
        )
    pool.shutdown()
    for row in comparison:
        print(row)


if __name__ == "__main__":
    main()
