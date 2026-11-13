# Shared paths for the benchmark scripts. Source from the repo root.
REPO="$(pwd)"
[[ -f "$REPO/benchmark/env.sh" ]] || { echo "Run from the alpasim repo root" >&2; return 1; }

NRE_ARTIFACTS="$REPO/data/nre-artifacts"
DRIVERS_DIR="$REPO/data/drivers"
NRE_CONTAINER="$REPO/containers/nre-26.04.sif"
# Model weights live under the project directory: the home quota cannot hold them.
export HF_HOME="$REPO/data/hf_cache"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

# Wizard overrides per benchmark arm. The Alpamayo-family arms share the
# 4-camera, 6-sample public_2601 comparison preset.
ALPAMAYO="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam"
FUSED="$ALPAMAYO +exp/sim=safety_validator_radar"
declare -A ARM_OVERRIDES=(
  [vavam]="driver=vavam"
  [alpamayo1_5]="$ALPAMAYO"
  [simforge_d1_5]="$ALPAMAYO driver.model.checkpoint_path=SIMFORGE-AI/SIMFORGE-D1.5"
  # Visual feature cache arms (src/driver/src/alpasim_driver/visual_cache).
  [alpamayo1_5_vcache_exact]="$ALPAMAYO +driver.model.visual_cache.enabled=true"
  [alpamayo1_5_vcache_neardup]="$ALPAMAYO +driver.model.visual_cache.enabled=true +driver.model.visual_cache.strategy=near_duplicate"
  # Safety validator arms (src/wizard/configs/exp/sim/safety_validator*.yaml),
  # all with the comparison preset's 6 candidates.
  [alpamayo1_5_safety_lidar]="$ALPAMAYO +exp/sim=safety_validator"
  [alpamayo1_5_safety_radar]="$FUSED runtime.simulation_config.render_lidar_for_driver=false"
  [alpamayo1_5_safety_fused]="$FUSED"
  # The vehicle ahead brakes hard to a stop (exp/scenario/lead_brake.yaml).
  [alpamayo1_5_leadbrake]="$ALPAMAYO +exp/scenario=lead_brake"
  [alpamayo1_5_safety_fused_leadbrake]="$FUSED +exp/scenario=lead_brake"
  # Baselines: emergency braking only on the model's single plan, and RSS
  # longitudinal safe distance on the fused sensors.
  [alpamayo1_5_aeb]="$ALPAMAYO +exp/sim=safety_validator driver.model.num_trajectory_samples=1 +driver.safety_validator.enable_stopping_distance=false +driver.safety_validator.enable_low_ttc=false"
  [alpamayo1_5_rss]="$FUSED +driver.safety_validator.rss_baseline=true"
  # VaVAM gives one candidate, so the validator can only slow it down.
  [vavam_safety_lidar]="driver=vavam +exp/sim=safety_validator"
  [vavam_safety_fused]="driver=vavam +exp/sim=safety_validator_radar"
  [vavam_leadbrake]="driver=vavam +exp/scenario=lead_brake"
  [vavam_safety_fused_leadbrake]="driver=vavam +exp/sim=safety_validator_radar +exp/scenario=lead_brake"
  # Ablations of the fused method.
  [abl_n1]="$FUSED driver.model.num_trajectory_samples=1"
  [abl_beta0]="$FUSED +driver.safety_validator.confidence_weight_power=0"
  [abl_beta1]="$FUSED +driver.safety_validator.confidence_weight_power=1"
  [abl_no_emg]="$FUSED +driver.safety_validator.enable_emergency=false"
  [abl_no_stopdist]="$FUSED +driver.safety_validator.enable_stopping_distance=false"
  [abl_no_lowttc]="$FUSED +driver.safety_validator.enable_low_ttc=false"
  [abl_no_rear]="$FUSED +driver.safety_validator.rear_prediction_horizon=0"
  # Weather arms: LiDAR and radar degraded to each dev scene's actual weather
  # (exp/weather/dev_scenes.yaml).  Fixed-sensor arms vs. the adaptive policy.
  [alpamayo1_5_safety_lidar_weather]="$ALPAMAYO +exp/sim=safety_validator +exp/weather=dev_scenes"
  [alpamayo1_5_safety_fused_weather]="$FUSED +exp/weather=dev_scenes"
  [alpamayo1_5_safety_adaptive_weather]="$ALPAMAYO +exp/sim=safety_validator_weather +exp/weather=dev_scenes"
  # Scenario labels only (logged and in videos; driving unchanged), to measure
  # their distribution and query cost against alpamayo1_5.
  [alpamayo1_5_scenario_labels]="$ALPAMAYO +exp/labels=scenario"
)
N_ROLLOUTS=${N_ROLLOUTS:-3}
# Where prepared arms live; benchmark/summarize.py reads benchmark/runs.
RUNS_ROOT=${RUNS_ROOT:-$REPO/benchmark/runs}

arm_overrides() {
  local arm=$1
  [[ -n "${ARM_OVERRIDES[$arm]:-}" ]] || {
    echo "Unknown arm '$arm'. Known: ${!ARM_OVERRIDES[*]}" >&2
    return 1
  }
  echo "${ARM_OVERRIDES[$arm]}"
}
