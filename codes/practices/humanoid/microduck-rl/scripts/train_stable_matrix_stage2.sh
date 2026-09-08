#!/usr/bin/env bash
# Extend the stage-1 policies through the task curricula.
#
# Stage 1 performs a common 500-iteration convergence screen.  This pass
# resumes the newest checkpoint for each task and trains to a task-specific
# curriculum milestone.  The milestones are intentionally explicit: they are
# not a claim that one scalar reward proves sim-to-real stability; the follow-up
# evaluator/rendering pass still checks survival, terminations and playback.

set -u

cd "$(dirname "$0")/.."

experiment="mode_matrix_stable"
root="logs/rsl_rl/${experiment}"
run_log_root="logs/${experiment}"
mkdir -p "$root" "$run_log_root"

# task-id | stage-1 slug (or fallback source run) | target total iteration | envs
tasks=(
  "Mjlab-Velocity-Flat-MicroDuck|velocity-flat-stage1|3000|64"
  "Mjlab-Velocity-Rough-MicroDuck|velocity-rough-stage1|3000|64"
  "Mjlab-VelStand-Flat-MicroDuck|velstand-flat-stage1|2500|64"
  "Mjlab-VelStand-Rough-MicroDuck|velstand-rough-stage1|2500|64"
  "Mjlab-StandUp-Flat-MicroDuck|standup-flat-stage1|4000|64"
  "Mjlab-StandUp-Rough-MicroDuck|standup-rough-stage1|4000|64"
  "Mjlab-SitStand-Flat-MicroDuck|sitstand-flat-stage1|2500|64"
  "Mjlab-SitStand-Rough-MicroDuck|sitstand-rough-stage1|2500|64"
  "Mjlab-GroundPick-Flat-MicroDuck|groundpick-flat-stage1|2000|64"
  "Mjlab-GroundPick-Rough-MicroDuck|groundpick-rough-stage1|2000|64"
  "Mjlab-BallKick-Flat-MicroDuck|ballkick-flat-stage1|1500|64"
  "Mjlab-Velocity-Flat-MicroDuck-Rollers|roller-velocity-stage1|5000|32"
  "Mjlab-Velocity-Swizzle-MicroDuck|roller-swizzle-stage1|3000|32"
  "Mjlab-RollerCrouch-Flat-MicroDuck|roller-crouch-stage1|1500|32"
  "Mjlab-RollerSlope-Flat-MicroDuck|roller-slope-stage1|1500|32"
  "Mjlab-RollerStandUp-Flat-MicroDuck|roller-standup-stage1|4000|32"
  "Mjlab-Spin-Flat-MicroDuck|spin-flat-stage1|3000|32"
  "Mjlab-Roulade-Flat-MicroDuck|roulade-flat-stage1|6000|64"
)

status_file="${run_log_root}/stage2_status.tsv"
printf 'task\tstatus\trun\tcheckpoint\tadditional_iterations\n' > "$status_file"

for spec in "${tasks[@]}"; do
  IFS='|' read -r task slug target envs <<< "$spec"
  base_slug="${slug%-stage1}"

  if [[ -n "${TASK_REGEX:-}" && ! "$task" =~ ${TASK_REGEX} ]]; then
    continue
  fi
  if [[ -n "${SKIP_TASK_REGEX:-}" && "$task" =~ ${SKIP_TASK_REGEX} ]]; then
    printf '%s\tskipped\t%s\t-\t-\n' "$task" "$slug" >> "$status_file"
    continue
  fi

  # Prefer a completed stage-2 run when this script is resumed.  Otherwise a
  # stage-1 run is named with a timestamp followed by _${slug}.  Select the
  # highest checkpoint number so a rerun resumes from the best completed stage
  # rather than from the original smoke model.
  checkpoint="$(find "$root" -mindepth 2 -maxdepth 2 -type f -path "*_${base_slug}-stage2/model_*.pt" -printf '%p\n' 2>/dev/null | awk -F/ '{print $(NF-1) "\t" $NF "\t" $0}' | sort -k2,2V | tail -1 | cut -f3-)"
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$(find "$root" -mindepth 2 -maxdepth 2 -type f -path "*_${slug}/model_*.pt" -printf '%p\n' 2>/dev/null | awk -F/ '{print $(NF-1) "\t" $NF "\t" $0}' | sort -k2,2V | tail -1 | cut -f3-)"
  fi

  # SitStand-Flat is allowed to use the higher-quality 256-env probe if stage
  # 1 was intentionally skipped while that probe was running.
  if [[ -z "$checkpoint" && "$task" == "Mjlab-SitStand-Flat-MicroDuck" ]]; then
    checkpoint="$(find logs/rsl_rl/mode_matrix_smoke -mindepth 2 -maxdepth 2 -type f -path '*_sitstand-flat-500/model_*.pt' -printf '%p\n' 2>/dev/null | sort -V | tail -1)"
  fi

  if [[ -z "$checkpoint" ]]; then
    echo "MISSING checkpoint for ${task} (${slug})" | tee "${run_log_root}/${slug}-stage2.log"
    printf '%s\tmissing\t%s\t-\t-\n' "$task" "$slug" >> "$status_file"
    continue
  fi

  checkpoint_name="${checkpoint##*/}"
  current="${checkpoint_name#model_}"
  current="${current%.pt}"
  # runner.learn() iterates from current_learning_iteration up to (but not
  # including) current + num_learning_iterations, then saves the final `it`.
  # Therefore reaching model_${target}.pt requires target-current+1 updates.
  if (( current >= target )); then
    printf '%s\talready-at-%s\t%s\t%s\t0\n' "$task" "$current" "${checkpoint%/*}" "$checkpoint_name" >> "$status_file"
    continue
  fi
  additional=$((target - current + 1))

  # Stage the checkpoint under the same experiment so mjlab's --load-run
  # resolver can load it, even when it originated from the smoke experiment.
  stage_run="stage2_source_${base_slug}"
  stage_path="${root}/${stage_run}"
  mkdir -p "$stage_path"
  cp "$checkpoint" "${stage_path}/${checkpoint_name}"
  task_log="${run_log_root}/${slug}-stage2.log"

  echo "[stable-matrix] ${task}: ${current} -> ${target} (+${additional}), ${envs} envs"
  set +e
  WANDB_MODE=offline uv run train "$task" \
    --env.scene.num-envs "$envs" \
    --env.seed 42 --agent.seed 42 \
    --agent.resume True \
    --agent.load-run "$stage_run" \
    --agent.load-checkpoint "$checkpoint_name" \
    --agent.max-iterations "$additional" \
    --agent.save-interval 250 \
    --agent.logger tensorboard \
    --agent.experiment-name "$experiment" \
    --agent.run-name "${base_slug}-stage2" \
    --agent.upload-model False > "$task_log" 2>&1
  code=$?
  set +e

  if [[ "$code" -eq 0 ]]; then
    final_ckpt="$(find "$root" -mindepth 2 -maxdepth 2 -type f -path "*_${base_slug}-stage2/model_*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    printf '%s\tok\t%s\t%s\t%s\n' "$task" "${final_ckpt%/*}" "${final_ckpt##*/}" "$additional" >> "$status_file"
    echo "[stable-matrix] ${task}: OK (${final_ckpt:-checkpoint not found})"
  else
    printf '%s\tfail(%s)\t%s\tunknown\t%s\n' "$task" "$code" "$slug" "$additional" >> "$status_file"
    echo "[stable-matrix] ${task}: FAILED with exit code ${code}; see ${task_log}"
  fi
done

echo "[stable-matrix] stage 2 complete; status: ${status_file}"
