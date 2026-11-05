#!/bin/bash
# 为 2026-09-08 重训完成的 3 个任务导出 play 视频并转 GIF
cd "$(dirname "$0")/.." || exit 1
OUT=logs/gifs; mkdir -p $OUT
declare -A JOBS=(
  [Mjlab-Velocity-Rough-MicroDuck]=logs/rsl_rl/velocity/2026-09-07_11-12-47_velocity
  [Mjlab-VelStand-Rough-MicroDuck]=logs/rsl_rl/velstand/2026-09-08_00-56-57_velstand
  [Mjlab-SitStand-Flat-MicroDuck]=logs/rsl_rl/microduck_sitstand/2026-09-08_08-40-55_microduck_sitstand
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
