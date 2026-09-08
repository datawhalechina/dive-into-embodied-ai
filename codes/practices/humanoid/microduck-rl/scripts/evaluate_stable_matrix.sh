#!/usr/bin/env bash
# Post-training evaluation for the stable matrix.

set -u

cd "$(dirname "$0")/.."

experiment="mode_matrix_stable"
root="logs/rsl_rl/${experiment}"
out_root="logs/${experiment}/videos"
mkdir -p "$out_root"

tasks=(
  "Mjlab-Velocity-Flat-MicroDuck|velocity-flat"
  "Mjlab-Velocity-Rough-MicroDuck|velocity-rough"
  "Mjlab-VelStand-Flat-MicroDuck|velstand-flat"
  "Mjlab-VelStand-Rough-MicroDuck|velstand-rough"
  "Mjlab-StandUp-Flat-MicroDuck|standup-flat"
  "Mjlab-StandUp-Rough-MicroDuck|standup-rough"
  "Mjlab-SitStand-Flat-MicroDuck|sitstand-flat"
  "Mjlab-SitStand-Rough-MicroDuck|sitstand-rough"
  "Mjlab-GroundPick-Flat-MicroDuck|groundpick-flat"
  "Mjlab-GroundPick-Rough-MicroDuck|groundpick-rough"
  "Mjlab-BallKick-Flat-MicroDuck|ballkick-flat"
  "Mjlab-Velocity-Flat-MicroDuck-Rollers|roller-velocity"
  "Mjlab-Velocity-Swizzle-MicroDuck|roller-swizzle"
  "Mjlab-RollerCrouch-Flat-MicroDuck|roller-crouch"
  "Mjlab-RollerSlope-Flat-MicroDuck|roller-slope"
  "Mjlab-RollerStandUp-Flat-MicroDuck|roller-standup"
  "Mjlab-Spin-Flat-MicroDuck|spin-flat"
  "Mjlab-Roulade-Flat-MicroDuck|roulade-flat"
)

status="logs/${experiment}/evaluation_status.tsv"
printf 'task\tstatus\tcheckpoint\n' > "$status"

for spec in "${tasks[@]}"; do
  IFS='|' read -r task slug <<< "$spec"
  if [[ -n "${TASK_REGEX:-}" && ! "$task" =~ ${TASK_REGEX} ]]; then
    continue
  fi

  checkpoint="$(find "$root" -mindepth 2 -maxdepth 2 -type f -path "*_${slug}-stage2/model_*.pt" -printf '%p\n' 2>/dev/null | sort -V | tail -1)"
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$(find "$root" -mindepth 2 -maxdepth 2 -type f -path "*_${slug}-stage1/model_*.pt" -printf '%p\n' 2>/dev/null | sort -V | tail -1)"
  fi
  if [[ -z "$checkpoint" ]]; then
    printf '%s\tmissing\t-\n' "$task" >> "$status"
    continue
  fi

  base="${out_root}/${slug}"
  log="${out_root}/${slug}.log"
  echo "[stable-matrix] evaluating ${task}: ${checkpoint}"
  set +e
  MUJOCO_GL=egl uv run python scripts/render_mjlab_checkpoint.py "$task" \
    --checkpoint "$checkpoint" \
    --mp4 "${base}.mp4" \
    --gif "${base}.gif" \
    --frames 200 --device cuda:0 \
    --lin-vel-x 0.0 --lin-vel-y 0.0 --ang-vel-z 0.0 > "$log" 2>&1
  code=$?
  set +e
  if [[ "$code" -eq 0 ]]; then
    printf '%s\tok\t%s\n' "$task" "$checkpoint" >> "$status"
  else
    printf '%s\tfail(%s)\t%s\n' "$task" "$code" "$checkpoint" >> "$status"
    echo "[stable-matrix] ${task} evaluation failed; see ${log}"
  fi
done

uv run python scripts/summarize_training_matrix.py \
  --root "$root" \
  --csv "logs/${experiment}/summary.csv" \
  --plot "logs/${experiment}/summary.webp"
echo "[stable-matrix] evaluation complete; status: ${status}"
