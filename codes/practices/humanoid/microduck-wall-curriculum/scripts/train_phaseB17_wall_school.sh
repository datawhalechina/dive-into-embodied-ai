#!/bin/bash
# Phase-B v17 (2026-10-20, 方向2: 放宽 flatness gate 允许斜向肩滚).
# B13-B16 frontier 全卡 90-140° 未过 180°。假设: 鸭子形貌翻不过头顶正上
# 方, 肩-头对角线滚物理更容易。改动: mdp flatness gate 30/60°→45/75°,
# head-top 严判撤销; 罚项 sagittal/flatness/lateral_vel 清零。warm 接 B14
# 最好 ckpt (B16 无增益, takeoff 回 B14 的 10/45°)。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.100
export DIVEWALL_WALL_DIST=0.40,0.65
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1
export DIVEWALL_PHASE_B17=1

LOAD_RUN=2026-10-19_07-56-52_microduck_roulade_divewall_run15
LOAD_CKPT=model_188473.pt
MAX_ITER=8000

LOG=logs/train_phaseB17_wall.log
for ENVS in 2048 1024; do
  echo "=== [Phase-B v17 shoulder-roll] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-B v17 shoulder-roll rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-B v17 训练结束 @ $(date) ==="

# 自动出 GIF — 低机位贴墙视角
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-B v17] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 3 --azimuth 95 --distance 2.0 \
  --video-width 1280 --video-height 720 --video-length 900 --num-envs 1 \
  --speed-overlay \
  > logs/gif_phaseB17_wall_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseB17_wall_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseB17_wall_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
