#!/bin/bash
# Stage-4 v3.6 (user 2026-10-12 GIF verdict: 加速向前→撞墙摔倒):
#   ① 墙继续降矮: DIVEWALL_WALL_SINK=0.09→0.105 → 有效墙高 12-10.5=1.5cm
#      (先体验"扑过去能过", 再逐步升回; 距离课程已验证此套路)
#   ② warm-start 自 v3.5 收官 model_38496 (跑→跳梯度桥已起效, takeoff/flight 刚冒头)
#   ③ 其余配方不动: takeoff 坡道/wall_lateral 1.8m/hop 6.0 等保持 v3.5
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.105
export DIVEWALL_POSTDIVE_PROB=0.25

LOAD_RUN=2026-10-12_00-58-25_microduck_roulade_divewall_run15
LOAD_CKPT=model_38496.pt
MAX_ITER=6000

LOG=logs/train_curric_stage4_v36.log
for ENVS in 2048 1024; do
  echo "=== [curric-stage4 v3.6] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER sink=$DIVEWALL_WALL_SINK @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage4 v3.6 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage4 v3.6 结束 @ $(date) ==="

# 自动出 GIF (标准机位)
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif stage4 v3.6] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 8 --azimuth 100 --distance 1.6 \
  --video-width 1280 --video-height 720 --video-length 600 --num-envs 1 \
  > logs/gif_stage4_v36_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/curric_stage4_v36_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/curric_stage4_v36_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
