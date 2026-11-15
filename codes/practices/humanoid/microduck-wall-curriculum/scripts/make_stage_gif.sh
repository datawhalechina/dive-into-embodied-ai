#!/bin/bash
# 课程链通用 GIF 生成器 — 在阶段脚本尾部调用 (继承该阶段全套 DIVEWALL_* 环境变量!)
# 用法: make_stage_gif.sh <gif名>   → logs/gifs/<gif名>.gif
# 注意: 取最新 run 的最新 ckpt, 必须紧跟该阶段训练结束调用。
set -u
NAME=${1:?usage: make_stage_gif.sh <gif名>}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/gifs
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-* 2>/dev/null | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "[gif error] 无 ckpt ($RUN_DIR)"; exit 1; }
echo "=== [gif $NAME] ckpt=$CKPT @ $(date +%T) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > "logs/gif_${NAME}.log" 2>&1 || { echo "[gif error] video_rollout 失败, 见 logs/gif_${NAME}.log"; exit 1; }
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "logs/gifs/${NAME}.gif" > /dev/null 2>&1 \
  && echo "=== [gif done] logs/gifs/${NAME}.gif ===" || echo "[gif error] ffmpeg 转换失败"
