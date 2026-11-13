#!/bin/bash
# Every CPU result in the supplementary material, from finished closed-loop
# rollouts (benchmark/runs*/<arm>*/rollouts/*/*/rollout.asl; see README.md,
# "Closed-loop rollouts").  Writes benchmark/final/ and, under each runs
# directory, the replay outputs validator_replay_final*.  Run from the repo root:
#
#   bash benchmark/reproduce_cpu.sh            # everything (several CPU-days)
#   bash benchmark/reproduce_cpu.sh tables     # only the tables, from the shipped CSVs
#
# WORKERS sets the processes per step (default 8).  The hazard test is
# deterministic: rerunning a configuration gives identical CSVs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 TOKIO_WORKER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
FINAL=benchmark/final
WORKERS=${WORKERS:-8}
mkdir -p "$FINAL/logs"
stamp() { echo "$(date '+%m-%d %H:%M') $*"; }

hazard() {  # tag, then hazard_stress.py arguments
  local tag=$1; shift
  uv run python benchmark/hazard_stress.py --workers "$WORKERS" "$@" \
    --output "$FINAL/hazard_$tag.csv" > "$FINAL/logs/hazard_$tag.log" 2>&1
  stamp "hazard $tag done"
}
replay() {  # runs_dir arm output_name, then replay_validator.py arguments
  local runs_dir=$1 arm=$2 name=$3; shift 3
  uv run python benchmark/replay_validator.py "$runs_dir" --arm "$arm" --workers "$WORKERS" "$@" \
    --output-name "$name" > "$FINAL/logs/replay_$name.log" 2>&1
  stamp "replay $name done"
}
counterfactual() {  # runs_dir arm output_name, then counterfactual_braking.py arguments
  local runs_dir=$1 arm=$2 name=$3; shift 3
  uv run python benchmark/counterfactual_braking.py "$runs_dir" "$name" --arm "$arm" --workers "$WORKERS" "$@" \
    > "$FINAL/logs/counterfactual_$name.log" 2>&1
  stamp "counterfactual $name done"
}

tables() {
  uv run python benchmark/supp_tables.py > "$FINAL/supp_tables.md"
  uv run python benchmark/final_tables.py > /dev/null  # writes $FINAL/tables.md
  stamp "tables written: $FINAL/supp_tables.md, $FINAL/tables.md"
}

if [[ ${1:-} == tables ]]; then tables; exit 0; fi

# Finished baseline rollouts (no validator): runs directory, arm, replay-name suffix.
SETS="benchmark/runs_smoke:alpamayo1_5_leadbrake:_braking_car
benchmark/runs_core:alpamayo1_5:
benchmark/runs:alpamayo1_5:_alpamayo1_5
benchmark/runs:alpamayo1_5_vcache_exact:_alpamayo1_5_vcache_exact
benchmark/runs:alpamayo1_5_vcache_neardup:_alpamayo1_5_vcache_neardup
benchmark/runs:simforge_d1_5:_simforge_d1_5
benchmark/runs:vavam:_vavam"
ALPAMAYO_SETS=$(grep -E ":alpamayo1_5(_vcache_[a-z]+)?:" <<< "$SETS" | grep -v braking_car)

TTC="--single-candidate --set enable_stopping_distance=false --set enable_low_ttc=false --set crossing_horizon_s=0"
RSS="--set rss_baseline=true"
declare -A SWEEP=(
  [beta0]="--set confidence_weight_power=0" [beta1]="--set confidence_weight_power=1"
  [horizon1]="--set crossing_horizon_s=1" [horizon2]="--set crossing_horizon_s=2"
  [horizon4]="--set crossing_horizon_s=4" [decel4]="--set max_decel=4" [decel8]="--set max_decel=8"
)
declare -A ROBUST=(
  [clutter0]="--radar false_alarms_per_scan=0" [clutter3]="--radar false_alarms_per_scan=3"
  [pd095]="--radar detection_probability=0.95" [pd080]="--radar detection_probability=0.8"
  [dropout30]="--lidar-dropout 0.3" [dropout60]="--lidar-dropout 0.6"
)

# 1. Hazard-injection test (Secs. 1-3, 5-6).
hazard main                                              # all four sensing conditions
hazard no_crossing --set crossing_horizon_s=0            # crossing-prediction ablation
hazard main_fused_detail --sensors fused                 # per-case reaction, deceleration, impact speed
# shellcheck disable=SC2086
hazard baseline_ttc --sensors fused $TTC
# shellcheck disable=SC2086
hazard baseline_rss --sensors fused $RSS
for tag in beta0 beta1 horizon1 horizon2 horizon4 decel4 decel8; do
  # shellcheck disable=SC2086
  hazard "sweep_$tag" --sensors fused --sources alpamayo_core alpamayo_dev ${SWEEP[$tag]}
done
for tag in clutter0 clutter3 pd095 pd080 dropout30 dropout60; do
  # shellcheck disable=SC2086
  hazard "robust_$tag" --sensors fused ${ROBUST[$tag]}
done

# 2. Replay of the logged rollouts: normal-driving interventions, selection tiers,
#    and the open-loop counterfactual, for every sensing condition.
while IFS=: read -r runs_dir arm suffix; do
  replay "$runs_dir" "$arm" "validator_replay_final$suffix"
  counterfactual "$runs_dir" "$arm" "validator_replay_final$suffix"
done <<< "$SETS"
# Baselines and the sensitivity sweep in normal Alpamayo 1.5 driving (fused).
for spec in "baseline_ttc:$TTC" "baseline_rss:$RSS" \
            "sweep_beta0:${SWEEP[beta0]}" "sweep_beta1:${SWEEP[beta1]}" \
            "sweep_horizon1:${SWEEP[horizon1]}" "sweep_horizon2:${SWEEP[horizon2]}" \
            "sweep_horizon4:${SWEEP[horizon4]}" "sweep_decel4:${SWEEP[decel4]}" \
            "sweep_decel8:${SWEEP[decel8]}"; do
  tag=${spec%%:*} args=${spec#*:}
  while IFS=: read -r runs_dir arm suffix; do
    # shellcheck disable=SC2086
    replay "$runs_dir" "$arm" "validator_replay_final_$tag$suffix" --sensors fused $args
  done <<< "$ALPAMAYO_SETS"
done
# Validating only the policy's top candidate (N = 1).
while IFS=: read -r runs_dir arm suffix; do
  replay "$runs_dir" "$arm" "validator_replay_final_n1$suffix" --sensors fused --single-candidate
  counterfactual "$runs_dir" "$arm" "validator_replay_final_n1$suffix" --sensors fused
done <<< "$ALPAMAYO_SETS"

# 3. Analyses over the replays.
uv run python benchmark/path_model_ablation.py      # $FINAL/path_model_ablation.csv
uv run python benchmark/normal_driving_interventions.py  # interventions by cause, reasoning check
uv run python benchmark/rear_collisions.py validator_replay_final  # $FINAL/rear_collisions.csv
stamp "analyses done"

tables
