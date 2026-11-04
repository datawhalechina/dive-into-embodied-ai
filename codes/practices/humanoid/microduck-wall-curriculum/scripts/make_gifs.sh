#!/bin/bash
# 为 9 个训练成功的任务导出 play 视频并转 GIF
cd "$(dirname "$0")/.." || exit 1
OUT=logs/gifs; mkdir -p $OUT
declare -A JOBS=(
  [Mjlab-Velocity-Flat-MicroDuck]=logs/rsl_rl/velocity/2026-09-04_00-13-56_velocity
  [Mjlab-Velocity-Flat-MicroDuck-Rollers]=logs/rsl_rl/velocity_rollers/2026-09-04_10-19-22_velocity_rollers
  [Mjlab-Velocity-Swizzle-MicroDuck]=logs/rsl_rl/velocity_swizzle/2026-09-04_21-10-05_velocity_swizzle
  [Mjlab-VelStand-Flat-MicroDuck]=logs/rsl_rl/velstand/2026-09-05_08-25-57_velstand
  [Mjlab-StandUp-Flat-MicroDuck]=logs/rsl_rl/microduck_stand/2026-09-05_12-45-20_microduck_stand
  [Mjlab-StandUp-Rough-MicroDuck]=logs/rsl_rl/microduck_stand/2026-09-05_16-02-30_microduck_stand
  [Mjlab-GroundPick-Flat-MicroDuck]=logs/rsl_rl/ground_pick/2026-09-05_23-28-49_ground_pick
  [Mjlab-BallKick-Flat-MicroDuck]=logs/rsl_rl/ball_kick_right/2026-09-06_04-29-23_ball_kick_right
  [Mjlab-Roulade-Flat-MicroDuck]=logs/rsl_rl/microduck_roulade/2026-09-06_06-45-03_microduck_roulade
)
for TASK in "${!JOBS[@]}"; do
  RUN="${JOBS[$TASK]}"
  CKPT=$(ls -t "$RUN"/model_*.pt | head -1)
  echo "=== [gif] $TASK  ckpt=$CKPT @ $(date) ==="
  uv run python scripts/video_rollout.py "$TASK" --checkpoint-file "$CKPT" \
    --video-length 300 --num-envs 1 --distance 0.8 --elevation -20 \
    --video-width 960 --video-height 540 > logs/gif_export_$(basename $TASK).log 2>&1
  MP4=$(ls -t "$RUN"/videos/play/*.mp4 2>/dev/null | head -1)
  if [ -n "$MP4" ]; then
    ffmpeg -y -i "$MP4" -vf "fps=24,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
      "$OUT/${TASK}.gif" > /dev/null 2>&1 && echo "=== [gif done] $TASK -> $OUT/${TASK}.gif ===" \
      || echo "=== [gif FAIL] ffmpeg $TASK ==="
  else
    echo "=== [gif FAIL] no mp4 for $TASK (see logs/gif_export_$(basename $TASK).log) ==="
  fi
done
echo "=== ALL GIF DONE @ $(date) ==="; ls -lh $OUT/
