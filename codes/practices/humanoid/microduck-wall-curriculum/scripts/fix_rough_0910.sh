#!/bin/bash
# 09-10: 1) SitStand-Rough GIF  2) GroundPick-Rough 重训 (nanfix cfg, 2048 envs)
cd "$(dirname "$0")/.." || exit 1
OUT=logs/gifs

# --- 1) SitStand-Rough GIF (ckpt from 09-09 nanfix run) ---
TASK=Mjlab-SitStand-Rough-MicroDuck
RUN=logs/rsl_rl/microduck_sitstand/2026-09-09_19-12-28_microduck_sitstand
CKPT=$(ls -t "$RUN"/model_*.pt | head -1)
echo "=== [gif] $TASK ckpt=$CKPT @ $(date) ==="
uv run python scripts/video_rollout.py "$TASK" --checkpoint-file "$CKPT" \
  --video-length 300 --num-envs 1 --distance 0.8 --elevation -20 \
  --video-width 960 --video-height 540 > logs/gif_export_SitStand-Rough.log 2>&1
MP4=$(ls -t "$RUN"/videos/play/*.mp4 2>/dev/null | head -1)
if [ -n "$MP4" ]; then
  ffmpeg -y -i "$MP4" -vf "fps=24,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    "$OUT/${TASK}.gif" > /dev/null 2>&1 && echo "=== [gif done] $TASK ===" || echo "=== [gif FAIL] ffmpeg ==="
else
  echo "=== [gif FAIL] no mp4, see logs/gif_export_SitStand-Rough.log ==="
fi

# --- 2) GroundPick-Rough 重训 ---
TASK=Mjlab-GroundPick-Rough-MicroDuck
if [ -f "logs/queue_done_${TASK}.done" ]; then echo "[skip] $TASK done"; else
  echo "=== [train start] $TASK @ $(date) ==="
  uv run train "$TASK" --env.scene.num-envs 2048 --agent.max_iterations 10000 \
    > "logs/train_${TASK}_nanfix_0910.log" 2>&1
  RC=$?
  echo "=== [train done rc=$RC] $TASK @ $(date) ==="
  [ $RC -eq 0 ] && touch "logs/queue_done_${TASK}.done" || echo "[FAIL] rc=$RC"
fi
echo "=== ALL DONE @ $(date) ==="
