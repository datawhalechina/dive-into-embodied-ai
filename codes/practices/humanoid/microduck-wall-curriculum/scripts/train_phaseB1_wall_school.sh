#!/bin/bash
# Phase-B v1 真墙学校 (user 2026-10-14 22:41): Phase-A 六版无墙苦练均值动作
# 仍是跑步 — 直接上墙: 有效墙高 0.5cm (WALL_SINK 0.115), stage-4 全配置
# (v3.8 减恐惧权重) + heading_hold 移植 + takeoff pitch 门 10→60。
# warm 接 Phase-A v6 (run 2026-10-14_20-53-51 / model_98486)。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.115
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=1

LOAD_RUN=2026-10-14_20-53-51_microduck_roulade_divewall_run15
LOAD_CKPT=model_98486.pt
MAX_ITER=6000

LOG=logs/train_phaseB1_wall.log
for ENVS in 2048 1024; do
  echo "=== [Phase-B v1 wall school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v1 wall school rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-B v1 wall school 结束 @ $(date) ==="

# 自动出 GIF — 广角机位, 墙全程可见
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v1] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 18 --azimuth 100 --distance 5.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB1_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB1_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB1_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
