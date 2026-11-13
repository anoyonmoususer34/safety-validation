#!/bin/bash
# Prepare and submit the closed-loop runs (see README.md), one
# phase at a time.  Run on a LOGIN node, from anywhere:
#   bash benchmark/submit.sh status            # every phase: done / running / ...
#   bash benchmark/submit.sh PHASE [--dry-run] # prepare + submit what is missing
#   bash benchmark/submit.sh PHASE --retry     # also redo arms whose job failed
#   bash benchmark/submit.sh PHASE --prepare-only  # download scenes + prepare, no jobs
#   ARMS="arm1 arm2" bash benchmark/submit.sh PHASE  # only some of the phase's arms
#
# A phase is a scene set (benchmark/scenes_SET.txt), a rollout count and its
# arms; each set's runs live in benchmark/runs_SET/ and are summarized with
#   uv run python benchmark/summarize.py benchmark/runs_SET
# A phase with PHASE_PARTS > 1 runs each arm as that many jobs, ARM__p1 ...,
# on consecutive slices of the scene list: short jobs start sooner on a busy
# queue.  summarize.py merges the parts back into one arm.
# Submitting again is safe: arms that are done, queued or running are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source benchmark/env.sh

declare -A PHASE_SET=(
  [smoke]=smoke [main]=core [hard]=hard [repeat]=repeat [leadbrake]=leadbrake
  [vavam]=large [vavam_hard]=hard [vavam_leadbrake]=leadbrake [ablations]=hard [weather]=dev
)
declare -A PHASE_ROLLOUTS=(
  [smoke]=1 [main]=1 [hard]=3 [repeat]=10 [leadbrake]=2 [vavam]=3 [vavam_hard]=3 [vavam_leadbrake]=2 [ablations]=3 [weather]=1
)
# Smoke runs drive 4 s instead of 20 s: enough to exercise every code path.
declare -A PHASE_EXTRA=([smoke]="runtime.simulation_config.n_sim_steps=40")
declare -A PHASE_PARTS=([main]=4 [hard]=5 [leadbrake]=4 [ablations]=5)
declare -A PHASE_ARMS=(
  # One arm per code path the later phases use (sensor mode, switches, RSS,
  # single candidate, VaVAM); the other arms differ only in config values.
  [smoke]="alpamayo1_5_safety_lidar alpamayo1_5_safety_radar alpamayo1_5_safety_fused alpamayo1_5_aeb alpamayo1_5_rss abl_n1 vavam_safety_lidar"
  # Table 2.
  [main]="alpamayo1_5 alpamayo1_5_safety_fused alpamayo1_5_safety_lidar alpamayo1_5_safety_radar alpamayo1_5_aeb alpamayo1_5_rss"
  # The highest front-risk scenes, several rollouts each: where the policy
  # causes collisions itself.  Other safety arms can be added with ARMS=.
  [hard]="alpamayo1_5 alpamayo1_5_safety_fused"
  # The baseline's own at-fault collision scene, many rollouts: how often the
  # collision recurs and how often the filter prevents it.
  [repeat]="alpamayo1_5 alpamayo1_5_safety_fused"
  # The vehicle ahead brakes hard (benchmark/select_lead_brake.py picks the scenes).
  [leadbrake]="alpamayo1_5_leadbrake alpamayo1_5_safety_fused_leadbrake"
  # Model-agnostic generalization.
  [vavam]="vavam vavam_safety_lidar vavam_safety_fused"
  # The hard and lead-brake comparisons with VaVAM, about 10x cheaper per rollout.
  [vavam_hard]="vavam vavam_safety_fused"
  [vavam_leadbrake]="vavam_leadbrake vavam_safety_fused_leadbrake"
  # Table 4.  Their baselines (alpamayo1_5, _safety_fused) on these scenes
  # come from the hard phase.
  [ablations]="abl_n1 abl_beta0 abl_beta1 abl_no_emg abl_no_stopdist abl_no_lowttc abl_no_rear"
  # A8 (optional): the dev set has the rain scenes exp/weather/dev_scenes.yaml labels.
  [weather]="alpamayo1_5_safety_lidar_weather alpamayo1_5_safety_fused_weather alpamayo1_5_safety_adaptive_weather"
)
PHASE_ORDER="smoke main hard repeat leadbrake vavam_hard vavam_leadbrake vavam ablations weather"
# Wall-clock minutes per rollout on one A40 with 4 rollouts in flight; the
# L40S measured about 16 for Alpamayo 1.5 and 0.5 for VaVAM; the A40 runs
# Alpamayo inference about 1.55x slower (25 min/rollout).  Generous on
# purpose: an arm that hits its time limit loses its aggregate results.
MIN_PER_ROLLOUT_ALPAMAYO=${MIN_PER_ROLLOUT_ALPAMAYO:-35}
MIN_PER_ROLLOUT_VAVAM=${MIN_PER_ROLLOUT_VAVAM:-3}
MAX_MINUTES=$((48 * 60))  # gpuA40x4 limit

usage() {
  echo "Usage: bash benchmark/submit.sh {status|$(echo $PHASE_ORDER | tr ' ' '|')} [--dry-run|--retry|--prepare-only]" >&2
  exit 1
}

job_name() { echo "bench_$1_$2"; }  # set, run

# The run directories of a phase's arm: ARM, or ARM__p1 ... ARM__pN.
arm_runs() {
  local phase=$1 arm=$2 parts=${PHASE_PARTS[$1]:-1} k
  if (( parts == 1 )); then echo "$arm"; return; fi
  for ((k = 1; k <= parts; k++)); do echo "${arm}__p$k"; done
}

# Scene list of one run: the whole set, or part K of PARTS consecutive slices.
run_scenes_file() {
  local phase=$1 run=$2
  local set=${PHASE_SET[$phase]} parts=${PHASE_PARTS[$1]:-1}
  local scenes_file="benchmark/scenes_$set.txt"
  if (( parts == 1 )); then echo "$scenes_file"; return; fi
  local k=${run##*__p} part_file="$REPO/benchmark/runs_$set/scenes_p${run##*__p}.txt"
  local n per
  n=$(grep -cvE '^(#|$)' "$scenes_file")
  per=$(( (n + parts - 1) / parts ))
  mkdir -p "$(dirname "$part_file")"
  { echo "# part $k of $parts of $scenes_file"
    grep -vE '^(#|$)' "$scenes_file" | sed -n "$(( (k - 1) * per + 1 )),$(( k * per ))p"
  } > "$part_file"
  echo "$part_file"
}

# done | running | pending | failed | prepared | missing
arm_state() {
  local set=$1 run=$2
  local dir="$REPO/benchmark/runs_$set/$run"
  local queued
  queued=$(squeue -h -u "$USER" -n "$(job_name "$set" "$run")" -o %T | head -1)
  if [[ -f "$dir/aggregate/results-summary.json" ]]; then echo done
  elif [[ "$queued" == RUNNING ]]; then echo running
  elif [[ -n "$queued" ]]; then echo pending
  elif [[ -f "$dir/benchmark-env.yaml" ]]; then echo failed  # a job started it
  elif prepare_complete "$dir"; then echo prepared
  else echo missing  # includes a prepare that failed or was killed partway
  fi
}

# prepare.sh rewrites the wizard's container paths as its last step.
prepare_complete() {
  local dir=$1
  [[ -f "$dir/generated-user-config-0.yaml" && -f "$dir/driver-config.yaml" ]] &&
    ! grep -q "/mnt/" "$dir"/generated-*.yaml "$dir/driver-config.yaml"
}

time_limit() {  # arm, scenes file, rollouts per scene -> sbatch -t value
  local arm=$1 scenes_file=$2 rollouts=$3
  local n_scenes per minutes
  n_scenes=$(grep -cvE '^(#|$)' "$scenes_file")
  per=$MIN_PER_ROLLOUT_ALPAMAYO
  [[ $arm == vavam* ]] && per=$MIN_PER_ROLLOUT_VAVAM
  # Rollouts run 4 at a time, so round up to whole batches of 4; plus 30 min
  # for service start-up and evaluation.
  minutes=$(( ( (n_scenes * rollouts + 3) / 4 * 4 ) * per + 30 ))
  if (( minutes > MAX_MINUTES )); then
    echo "WARNING: $arm needs ~$((minutes / 60)) h for $n_scenes scenes; capped at 48 h. Split the scene set." >&2
    minutes=$MAX_MINUTES
  fi
  printf '%d:%02d:00' $((minutes / 60)) $((minutes % 60))
}

status() {
  local phase set arm run
  for phase in $PHASE_ORDER; do
    set=${PHASE_SET[$phase]}
    echo "== $phase (benchmark/scenes_$set.txt, ${PHASE_ROLLOUTS[$phase]} rollout(s)/scene)"
    [[ -f "benchmark/scenes_$set.txt" ]] || { echo "   scene list not created yet"; continue; }
    for arm in ${PHASE_ARMS[$phase]}; do
      for run in $(arm_runs "$phase" "$arm"); do
        printf '   %-38s %s\n' "$run" "$(arm_state "$set" "$run")"
      done
    done
  done
}

submit_phase() {
  local phase=$1 mode=$2
  local set=${PHASE_SET[$phase]} rollouts=${PHASE_ROLLOUTS[$phase]}
  local scenes_file="benchmark/scenes_$set.txt"
  local runs_root="$REPO/benchmark/runs_$set"
  [[ -f "$scenes_file" ]] || { echo "Missing $scenes_file; create the scene list first." >&2; exit 1; }
  local arms=${ARMS:-${PHASE_ARMS[$phase]}}
  # Fail before submitting anything if a phase names an unknown arm.
  local arm
  for arm in $arms; do arm_overrides "$arm" > /dev/null; done

  for arm in $arms; do
    local run
    for run in $(arm_runs "$phase" "$arm"); do
      local state limit run_scenes
      state=$(arm_state "$set" "$run")
      case $state in
        done|running|pending)
          echo "$run: $state, skipping"; continue ;;
        prepared)
          if [[ $mode == prepare-only ]]; then echo "$run: prepared"; continue; fi ;;
        failed)
          if [[ $mode != retry ]]; then
            echo "$run: FAILED (see benchmark/logs/$(job_name "$set" "$run")_*.err); rerun with --retry"
            continue
          fi
          local failed_dir="$REPO/benchmark/failed_runs/$set/${run}_$(date +%Y%m%d-%H%M%S)"
          echo "$run: moving the failed run to $failed_dir"
          [[ $mode == dry-run ]] || { mkdir -p "$(dirname "$failed_dir")"; mv "$runs_root/$run" "$failed_dir"; }
          state=missing ;;
      esac
      run_scenes=$(run_scenes_file "$phase" "$run")
      limit=$(time_limit "$arm" "$run_scenes" "$rollouts")
      if [[ $mode == dry-run ]]; then
        echo "$run: would $([[ $state == missing ]] && echo 'prepare and ')submit (-t $limit)"
        continue
      fi
      if [[ $state == missing ]]; then
        # A prepare that was killed (e.g. the login session dropped while scenes
        # downloaded) leaves a directory prepare.sh refuses to reuse.
        rm -rf "${runs_root:?}/$run"
        echo "$run: preparing..."
        RUNS_ROOT=$runs_root RUN_NAME=$run N_ROLLOUTS=$rollouts EXTRA_OVERRIDES=${PHASE_EXTRA[$phase]:-} \
          bash benchmark/prepare.sh "$arm" "$run_scenes" \
          > "benchmark/logs/prepare_${set}_${run}.log" 2>&1 || {
          echo "$run: prepare FAILED, see benchmark/logs/prepare_${set}_${run}.log" >&2
          continue
        }
      fi
      if [[ $mode == prepare-only ]]; then echo "$run: prepared"; continue; fi
      local job_id
      job_id=$(sbatch --parsable -J "$(job_name "$set" "$run")" -t "$limit" \
        benchmark/run_benchmark.sbatch "$run" "$runs_root")
      echo "$run: submitted job $job_id (-t $limit)"
    done
  done
}

mkdir -p benchmark/logs
case ${1:-} in
  status) status ;;
  smoke|main|hard|repeat|leadbrake|vavam_hard|vavam_leadbrake|vavam|ablations|weather)
    mode=submit
    case ${2:-} in
      --dry-run) mode=dry-run ;;
      --retry) mode=retry ;;
      --prepare-only) mode=prepare-only ;;
      "") ;;
      *) usage ;;
    esac
    submit_phase "$1" "$mode" ;;
  *) usage ;;
esac
