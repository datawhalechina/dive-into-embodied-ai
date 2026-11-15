#!/bin/bash
# Phase-B v6 2cm real-geometry wall (2026-10-17 sink bug fix: geom now truly
# shrinks, H = 0.12 - sink). warm 接 B5 (model_116483)。其余同 B5:
# 近墙 0.40-0.65m / takeoff 0-50° / runup 8 / heading_hold。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.100
export DIVEWALL_WALL_DIST=0.40,0.65
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1

LOAD_RUN=2026-10-16_23-09-27_microduck_roulade_divewall_run15
LOAD_CKPT=model_116483.pt
MAX_ITER=6000

LOG=logs/train_phaseB6_wall.log
for ENVS in 2048 1024; do
  echo "=== [Phase-B v6 2cm real-geom wall] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v6 2cm real-geom wall rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-B v6 2cm real-geom wall 训练结束 @ $(date) ==="

# 自动出 GIF — 低机位贴墙视角 (user-approved 2026-10-17)
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v6 2cm] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB6_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB6_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB6_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
