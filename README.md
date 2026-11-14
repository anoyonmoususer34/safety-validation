# Safety Validation for End-to-End Autonomous Driving

Code, configurations and results for the paper *Safety Validation for End-to-End
Autonomous Driving* (anonymous submission, CVPR 2027) and its supplementary material.

The paper adds a runtime safety validator to end-to-end driving policies. The
validator checks each trajectory the policy proposes against obstacles sensed by
LiDAR and radar, drives a safe candidate when there is one, and otherwise slows
the ego vehicle. Every experiment runs in [AlpaSim](https://github.com/NVlabs/alpasim),
and this repository is a modified copy of AlpaSim (Apache 2.0; see `NOTICE`).

Every table in the supplementary material can be regenerated from the CSV files in
this repository in a few minutes on a CPU (see [Regenerating the tables](#regenerating-the-tables)).
Regenerating the CSV files themselves requires the closed-loop simulator (GPU) and
then several CPU-days of analysis (see [Reproducing from scratch](#reproducing-from-scratch)).

## Contents

| Path | What it is |
|---|---|
| `src/driver/src/alpasim_driver/safety_validator.py` | The validator: constraints C1–C4, the rear check, per-obstacle assessment, decision trace |
| `src/driver/src/alpasim_driver/safety_filter.py` | Candidate selection (tiers), speed override, sensing entry point |
| `src/driver/src/alpasim_driver/lidar_obstacle_extractor.py` | LiDAR clustering and tracking |
| `src/driver/src/alpasim_driver/radar_fusion.py` | Radar association and LiDAR + radar fusion |
| `src/driver/src/alpasim_driver/physics.py` | Path interpolation, crossing prediction, stopping distance |
| `src/driver/src/alpasim_driver/schema.py` | `SafetyValidatorConfig`: every parameter and its default |
| `src/runtime/alpasim_runtime/radar.py` | Simulated radar (two beams, noise, detection probability, clutter) |
| `src/runtime/alpasim_runtime/lead_brake.py` | Closed-loop lead-vehicle braking scenario |
| `src/runtime/alpasim_runtime/sensor_weather.py` | LiDAR and radar weather degradation |
| `src/utils/alpasim_utils/{lidar,radar,sensor}_payload.py` | Sensor payload formats between runtime and driver |
| `src/lidar_perception/` | LiDAR perception pipeline for real PhysicalAI-AV LiDAR |
| `src/wizard/configs/exp/sim/safety_validator*.yaml` | Wizard configurations that enable the validator (LiDAR, fused, weather-adaptive) |
| `src/wizard/configs/exp/scenario/` | Scenario configurations (lead braking) |
| `benchmark/` | Experiment scripts (below) |
| `benchmark/final/` | Final results: per-injection hazard-test CSVs and analysis outputs |
| `benchmark/runs*/` | Per-run configurations, per-rollout metrics, and the replay CSVs of the validator over the logged rollouts |
| `benchmark/radar_config.yaml` | Radar models used in the closed-loop runs, the replays and the hazard test |
| `docs/ALPASIM_README.md` | The original AlpaSim README; `docs/` has AlpaSim's own documentation |

### Experiment scripts

| Script | Purpose |
|---|---|
| `benchmark/hazard_stress.py` | Hazard-injection test: injects 10 hazard variants into each logged rollout and runs the validator closed loop on the policy's logged plans |
| `benchmark/replay_validator.py` | Replays the validator step by step over logged rollouts (normal-driving interventions, selection tiers, baselines, sweep) |
| `benchmark/counterfactual_braking.py` | Open-loop counterfactual for the replays: does following the validator avoid the logged collision, and which new ones occur |
| `benchmark/path_model_ablation.py` | Path model for obstacle relevance (straight, arc, interpolated) |
| `benchmark/normal_driving_interventions.py` | Interventions in normal driving by cause; reasoning text vs. intervention |
| `benchmark/rear_collisions.py` | Logged rear-end collisions and whether the rear check flagged them |
| `benchmark/supp_tables.py` | Supplementary Secs. 1–2, 4, 6, 7 and the robustness table → `benchmark/final/supp_tables.md` |
| `benchmark/final_tables.py` | Baselines, sweep, McNemar test, tiers, N = 1, constraints, comfort, misses → `benchmark/final/tables.md` |
| `benchmark/summarize.py` | Closed-loop driving scores, latency and GPU memory per policy → `benchmark/results.csv` |
| `benchmark/row_from_hazard.py` | One summary line for any hazard-test CSV |
| `benchmark/make_media.py` | Figures and videos of the supplementary (needs the raw logs and rendered videos, not included) |
| `benchmark/reproduce_cpu.sh` | Runs every CPU step above, in order |
| `benchmark/submit.sh`, `prepare.sh`, `run_benchmark.sbatch`, `env.sh` | Closed-loop runs on a SLURM cluster |
| `benchmark/select_scenes.py`, `screen_scenes.py`, `select_lead_brake.py` | Scene selection (scene lists in `benchmark/scenes_*.txt`) |

## Where each result comes from

Section and table names follow the supplementary material. "Tables" means the table
is printed by `supp_tables.py` (`supp_tables.md`) or `final_tables.py` (`tables.md`).

| Supplementary | Data | Printed by |
|---|---|---|
| Sec. 1, protocol counts (114 rollouts, 1,140 injections, 962 applicable) | `final/hazard_main.csv` | `supp_tables.py` |
| Prevention by hazard; per-policy tables (Alpamayo 1.5, SIMFORGE-D1.5, VaVAM); by speed | `final/hazard_main.csv` | `supp_tables.py` |
| Occluded vs. visible pedestrians | `final/hazard_main_fused_detail.csv` | `final_tables.py` |
| Crossing-prediction ablation | `final/hazard_main.csv`, `final/hazard_no_crossing.csv` | `final_tables.py` |
| Baselines (TTC-only, RSS), all policies and Alpamayo 1.5 | `final/hazard_baseline_{ttc,rss}.csv`; normal driving: `runs*/validator_replay_final{,_baseline_ttc,_baseline_rss}*/` | `final_tables.py` |
| McNemar test, ours vs. RSS | `final/hazard_main.csv`, `final/hazard_baseline_rss.csv` | `final_tables.py` |
| Sensitivity sweep (β, crossing horizon, deceleration) | `final/hazard_sweep_*.csv`; `runs*/validator_replay_final_sweep_*/` | `final_tables.py` |
| Sensor robustness (radar clutter, p_d, LiDAR dropout) | `final/hazard_robust_*.csv` | `supp_tables.py` |
| Selection tiers; N = 1 vs. N = 6 | `runs*/validator_replay_final{,_n1}*/` | `final_tables.py` |
| Path model for obstacle relevance | `final/path_model_ablation.csv` | `path_model_ablation.py` |
| Reasoning vs. intervention; interventions in normal driving by cause | `final/normal_driving_interventions.csv` | `normal_driving_interventions.py` |
| First constraint to fire; commanded deceleration | `final/hazard_main_fused_detail.csv`, `runs*/validator_replay_final*/` | `final_tables.py` |
| Rear-end outcomes (recorded and reacting followers) | `final/hazard_main.csv` | `supp_tables.py` |
| Collisions not prevented (braking- vs. sensing-limited, timing, impact speed) | `final/hazard_main.csv`, `final/hazard_main_fused_detail.csv` | `supp_tables.py`, `final_tables.py` |
| Design errors found by replay | before: `runs_core/validator_replay_before/` (5,954 steps); after: `runs_core/validator_replay_final/`, `final/rear_collisions.csv` | `replay_validator.py` output |
| Validator decisions by sensing condition (33 rollouts) | `runs_core/validator_replay/rollouts_*.csv` | `replay_validator.py` output |
| Policy comparison (scores, latency, GPU memory) | `runs/{alpamayo1_5,simforge_d1_5,vavam}/` (`aggregate/`, `driver/latency.jsonl`) | `summarize.py` → `results.csv` |
| Weather study | `runs_core/validator_replay/*_policy.csv`, `weather_label_eval.json` | `replay_validator.py`, `weather_label_eval.py` |
| Decision-trace example | `final/trace_example.json` | — |

`runs_core/validator_replay_before/` and `runs_core/validator_replay/` were produced
by earlier versions of the validator, before the fixes described in the
supplementary material. They are included as the evidence for those tables; the
code in this repository is the final version.

### Result files

**Hazard test** (`benchmark/final/hazard_*.csv`): one row per (rollout, hazard,
sensing condition). Key columns: `source`, `model`, `scene`, `rollout`, `hazard`,
`applicable`, `sensor` (`gt` = perfect perception, `radar`, `lidar`, `fused`),
`ego_speed_at_start_mps`, `baseline_hits_hazard`, `filtered_hits_hazard` (with the
validator), `first_override_s`, `overrides`, `baseline_impact_speed_mps`,
`filtered_impact_speed_mps`, and `new_other_collision[_react_1s|_react_1.5s]` /
`new_other_from_behind[...]` for new collisions under recorded followers or
followers that react after 1.0 s / 1.5 s. A collision is *prevented* when
`applicable & baseline_hits_hazard & ~filtered_hits_hazard`.
`hazard_main_fused_detail.csv` adds `hazard_onset_s`, `first_reaction_s`,
`first_reaction_constraints` and `commanded_decels`.

**Replays** (`benchmark/runs*/validator_replay*/`): `steps_<sensor>.csv` has one
row per policy step (`reason`, `switched`, `safe_speed_mps`, `model_violations`,
`violations`, closest obstacle); `rollouts_<sensor>.csv` one row per rollout;
`counterfactual_<sensor>.csv` the open-loop counterfactual per rollout.

**Closed-loop runs** (`benchmark/runs*/<arm>/`): the resolved configurations
(`generated-user-config-0.yaml`, `driver-config.yaml`, `wizard-config.yaml`, ...),
the GPU and software of the run (`benchmark-env.yaml`), and per rollout
`rollouts/<scene>/<rollout>/metrics.parquet`. Repository paths in the
configurations are written as `<repo>`.

## Setup

Requires Linux, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
source setup_local_env.sh      # creates .venv and installs every package (uv sync)
uv run pytest src/driver src/runtime/tests src/utils/tests src/wizard/tests
```

AlpaSim's own setup is described in `docs/ONBOARDING.md`.

## Regenerating the tables

From the repository root, with the shipped CSVs:

```bash
bash benchmark/reproduce_cpu.sh tables   # benchmark/final/supp_tables.md and tables.md
uv run python benchmark/summarize.py     # benchmark/results.csv (policy comparison)
uv run python benchmark/row_from_hazard.py benchmark/final/hazard_robust_pd080.csv
```

## Reproducing from scratch

### 1. Closed-loop rollouts (GPU)

Each run uses one 48 GB GPU, 16 CPU cores and 62 GB of memory; the job script
starts the neural renderer, physics, controller and driver on that GPU. Alpamayo 1.5
takes about 25 minutes per 20 s rollout on an NVIDIA A40 (about 16 on an L40S);
VaVAM about 3. `benchmark-env.yaml` in each run directory records the GPU used.

Assets (not included):

- scenes: the PhysicalAI-AV NuRec scenes listed in `benchmark/scenes_*.txt`, downloaded
  by the wizard from Hugging Face (gated; set `HF_TOKEN`);
- the NuRec renderer container, at `containers/nre-26.04.sif` (see `docs/ONBOARDING.md`);
- policy weights: `bash benchmark/download_weights.sh` (Alpamayo 1.5, SIMFORGE-D1.5)
  and `bash data/download_vavam_assets.sh` (VaVAM).


```bash
bash benchmark/submit.sh main         # 42 core scenes, 1 rollout: benchmark/runs_core/
bash benchmark/submit.sh leadbrake    # closed-loop lead braking: benchmark/runs_leadbrake/
bash benchmark/submit.sh status
# 8 dev scenes x 3 rollouts for each policy: benchmark/runs/
for arm in alpamayo1_5 alpamayo1_5_vcache_exact alpamayo1_5_vcache_neardup simforge_d1_5 vavam; do
  RUNS_ROOT=$PWD/benchmark/runs N_ROLLOUTS=3 bash benchmark/prepare.sh $arm benchmark/scenes_dev.txt
  sbatch benchmark/run_benchmark.sbatch $arm $PWD/benchmark/runs
done
```

The arms (Hydra overrides per configuration) are defined in `benchmark/env.sh`; the
validator arms add `+exp/sim=safety_validator` (LiDAR) or
`+exp/sim=safety_validator_radar` (LiDAR + radar). The hazard test and replays use
only the baseline runs, without the validator (`alpamayo1_5`, `simforge_d1_5`,
`vavam` and the two `alpamayo1_5_vcache_*` arms, which run Alpamayo 1.5 with a
visual-feature cache).

### 2. Hazard test, replays and analyses (CPU)

```bash
bash benchmark/reproduce_cpu.sh     # WORKERS=8 by default
```

This reruns every hazard-test configuration, replay and analysis, and then the
tables. It reads only the logged rollouts (`rollout.asl`), so it needs no GPU. The
hazard test is deterministic: running the same configuration twice gives identical
CSVs. With 8 workers each hazard-test configuration takes several hours; the whole
script takes several days.

## Not included

- Raw rollout logs (`rollout.asl`, about 48 GB) and rendered camera videos: they
  contain the PhysicalAI-AV scene data. Rerun step 1 to regenerate them.
- The supplementary videos, which are provided separately.
- Policy weights, scenes and the renderer container (see step 1).

## License

Apache License 2.0 (see `LICENSE`). This repository modifies AlpaSim; see `NOTICE`.

## LLM Usage Statement 

We used LLMs to help organize the code and format this README.md. All of the output has been verified by the human authors. 
