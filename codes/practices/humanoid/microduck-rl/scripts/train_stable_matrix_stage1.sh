#!/usr/bin/env bash
# Train every public MicroDuck task beyond the 50-iteration smoke check.
#
# The source checkpoints are the independent smoke policies already produced
# by this project.  They are copied into a staging run because mjlab resolves
# --agent.load-run relative to the selected experiment directory.  Each task
# keeps its own actor/critic/optimizer state; no weights are shared between
# modes.
#
# This first pass deliberately uses 500 additional PPO iterations and 64
# environments per task.  On the 4 GiB CUDA 12.2 development machine this is
# a useful convergence screen (roughly 0.8M transitions per task), after
# which failing modes can be extended without retraining the others.

set -u

cd "$(dirname "$0")/.."

experiment="mode_matrix_stable"
stage_root="logs/rsl_rl/${experiment}"
run_log_root="logs/${experiment}"
mkdir -p "$stage_root" "$run_log_root"

# task-id | source experiment | source run | source checkpoint | envs | extra iterations | run slug
tasks=(
  "Mjlab-Velocity-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-16-23_Velocity_Flat_MicroDuck|model_49.pt|64|500|velocity-flat-stage1"
  "Mjlab-Velocity-Rough-MicroDuck|mode_matrix_smoke|2026-09-03_14-18-26_Velocity_Rough_MicroDuck|model_49.pt|64|500|velocity-rough-stage1"
  "Mjlab-VelStand-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-21-43_VelStand_Flat_MicroDuck|model_49.pt|64|500|velstand-flat-stage1"
  "Mjlab-VelStand-Rough-MicroDuck|mode_matrix_smoke|2026-09-03_14-23-51_VelStand_Rough_MicroDuck|model_49.pt|64|500|velstand-rough-stage1"
  "Mjlab-StandUp-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-15-24_standup-flat-smoke|model_9.pt|64|500|standup-flat-stage1"
  "Mjlab-StandUp-Rough-MicroDuck|mode_matrix_smoke|2026-09-03_14-26-49_StandUp_Rough_MicroDuck|model_49.pt|64|500|standup-rough-stage1"
  "Mjlab-SitStand-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-29-40_SitStand_Flat_MicroDuck|model_49.pt|64|500|sitstand-flat-stage1"
  "Mjlab-SitStand-Rough-MicroDuck|mode_matrix_smoke|2026-09-03_14-31-43_SitStand_Rough_MicroDuck|model_49.pt|64|500|sitstand-rough-stage1"
  "Mjlab-GroundPick-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-34-29_GroundPick_Flat_MicroDuck|model_49.pt|64|500|groundpick-flat-stage1"
  "Mjlab-GroundPick-Rough-MicroDuck|mode_matrix_smoke|2026-09-03_14-36-51_GroundPick_Rough_MicroDuck|model_49.pt|64|500|groundpick-rough-stage1"
  "Mjlab-BallKick-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-39-37_BallKick_Flat_MicroDuck|model_49.pt|64|500|ballkick-flat-stage1"
  "Mjlab-Velocity-Flat-MicroDuck-Rollers|mode_matrix_smoke|2026-09-03_14-42-20_Velocity_Flat_MicroDuck_Rollers|model_49.pt|64|500|roller-velocity-stage1"
  "Mjlab-Velocity-Swizzle-MicroDuck|mode_matrix_smoke|2026-09-03_14-44-54_Velocity_Swizzle_MicroDuck|model_49.pt|64|500|roller-swizzle-stage1"
  "Mjlab-RollerCrouch-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-47-11_RollerCrouch_Flat_MicroDuck|model_49.pt|64|500|roller-crouch-stage1"
  "Mjlab-RollerSlope-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-49-32_RollerSlope_Flat_MicroDuck|model_49.pt|64|500|roller-slope-stage1"
  "Mjlab-RollerStandUp-Flat-MicroDuck|mode_matrix_smoke_fixed|2026-09-03_15-03-07_RollerStandUp_Flat_MicroDuck|model_49.pt|32|500|roller-standup-stage1"
  "Mjlab-Spin-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-52-21_Spin_Flat_MicroDuck|model_49.pt|64|500|spin-flat-stage1"
  "Mjlab-Roulade-Flat-MicroDuck|mode_matrix_smoke|2026-09-03_14-54-40_Roulade_Flat_MicroDuck|model_49.pt|64|500|roulade-flat-stage1"
)

status_file="${run_log_root}/stage1_status.tsv"
printf 'task\tstatus\trun\tcheckpoint\n' > "$status_file"

for spec in "${tasks[@]}"; do
  IFS='|' read -r task source_experiment source_run source_checkpoint envs iterations slug <<< "$spec"

  if [[ -n "${TASK_REGEX:-}" && ! "$task" =~ ${TASK_REGEX} ]]; then
    continue
  fi
  if [[ -n "${SKIP_TASK_REGEX:-}" && "$task" =~ ${SKIP_TASK_REGEX} ]]; then
    printf '%s\tskipped\t%s\t%s\n' "$task" "$slug" "$source_checkpoint" >> "$status_file"
    continue
  fi

  source_path="logs/rsl_rl/${source_experiment}/${source_run}/${source_checkpoint}"
  task_log="${run_log_root}/${slug}.log"

  if [[ ! -f "$source_path" ]]; then
    echo "MISSING ${source_path}" | tee "$task_log"
    printf '%s\tmissing\t%s\t%s\n' "$task" "$slug" "$source_checkpoint" >> "$status_file"
    continue
  fi

  # Make the stage restartable.  A clean shutdown/reboot can interrupt a
  # long task before its next save interval; on the next invocation, resume
  # from the highest checkpoint already produced for this slug.  Staging
  # directories also match this search and are harmless model_49 fallbacks.
  source_iter="${source_checkpoint#model_}"
  source_iter="${source_iter%.pt}"
  desired_final=$((source_iter + iterations - 1))
  latest="$(find "${stage_root}" -mindepth 2 -maxdepth 2 -type f -path "*_${slug}/model_*.pt" -printf '%p\n' 2>/dev/null | awk -F/ '{print $NF "\t" $0}' | sort -k1,1V | tail -1 | cut -f2-)"
  if [[ -n "$latest" ]]; then
    latest_name="${latest##*/}"
    latest_iter="${latest_name#model_}"
    latest_iter="${latest_iter%.pt}"
  else
    latest="$source_path"
    latest_name="$source_checkpoint"
    latest_iter="$source_iter"
  fi

  if (( latest_iter >= desired_final )); then
    printf '%s\tok\t%s\t%s\n' "$task" "${latest%/*}" "$latest_name" >> "$status_file"
    echo "[stable-matrix] ${task}: already complete (${latest_name})"
    continue
  fi

  additional=$((desired_final - latest_iter + 1))
  stage_run="stage_resume_${slug}"
  stage_path="${stage_root}/${stage_run}"
  mkdir -p "$stage_path"
  cp "$latest" "${stage_path}/${latest_name}"

  echo "[stable-matrix] ${task}: resume ${latest_name} -> ${desired_final} (+${additional}), ${envs} envs"
  set +e
  WANDB_MODE=offline uv run train "$task" \
    --env.scene.num-envs "$envs" \
    --env.seed 42 --agent.seed 42 \
    --agent.resume True \
    --agent.load-run "$stage_run" \
    --agent.load-checkpoint "$latest_name" \
    --agent.max-iterations "$additional" \
    --agent.save-interval 100 \
    --agent.logger tensorboard \
    --agent.experiment-name "$experiment" \
    --agent.run-name "$slug" \
    --agent.upload-model False > "$task_log" 2>&1
  code=$?
  set +e

  if [[ "$code" -eq 0 ]]; then
    final_ckpt="$(find "${stage_root}" -mindepth 2 -maxdepth 2 -type f -path "*_${slug}/model_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    printf '%s\tok\t%s\t%s\n' "$task" "$slug" "${final_ckpt:-unknown}" >> "$status_file"
    echo "[stable-matrix] ${task}: OK (${final_ckpt:-checkpoint not found})"
  else
    printf '%s\tfail(%s)\t%s\tunknown\n' "$task" "$code" "$slug" >> "$status_file"
    echo "[stable-matrix] ${task}: FAILED with exit code ${code}; see ${task_log}"
  fi
done

echo "[stable-matrix] complete; status: ${status_file}"
