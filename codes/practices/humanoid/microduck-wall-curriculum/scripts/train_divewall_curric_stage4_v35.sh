#!/bin/bash
# Stage-4 v3 (user 2026-10-11 GIF verdict: 无助跑直接翻滚 + 倒地蹬脚):
#   ① warm-start 自 Stage-3 v7e 收官 (会跑会前扑、没接触过墙 —— 最干净血统;
#      NOT stage4-v2, 它已学会"走到墙边扒墙刷 push 分")
#   ② env_cfg stage-4 分支: 继承 v7 跑步配方降权副本 (track/pose/upright/air_time)
#   ③ 墙距 0.85-1.2m → 2.5-3.5m (还助跑空间)
#   ④ DIVEWALL_WALL_SINK=0.09 → 有效墙高 12-9=3cm (user 拍板: 先 3cm)
#   ⑤ wall_push vx 硬门 0.3 / premature_roll vx 门控 0.2 (堵零速扒墙翻滚)
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.09
export DIVEWALL_POSTDIVE_PROB=0.25

LOAD_RUN=2026-10-11_18-14-02_microduck_roulade_divewall_run15
LOAD_CKPT=model_32497.pt
MAX_ITER=6000

LOG=logs/train_curric_stage4_v35.log
for ENVS in 2048 1024; do
  echo "=== [curric-stage4 v3.5] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER sink=$DIVEWALL_WALL_SINK @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage4 v3.5 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage4 v3.5 结束 @ $(date) ==="

# 自动出 GIF (标准机位)
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif stage4 v3.5] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 8 --azimuth 100 --distance 1.6 \
  --video-width 1280 --video-height 720 --video-length 600 --num-envs 1 \
  > logs/gif_stage4_v35_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/curric_stage4_v35_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/curric_stage4_v35_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
