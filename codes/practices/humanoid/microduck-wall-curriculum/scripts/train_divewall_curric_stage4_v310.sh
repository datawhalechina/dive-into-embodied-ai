#!/bin/bash
# Stage-4 v3.10 (user 2026-10-13 GIF verdict: 0.5cm 墙仍撞墙挂头 → 墙高不是唯一变量):
#   ① 对照实验: DIVEWALL_WALL_SINK=0.12 → 有效墙高 0cm (纯平地)
#      目的: 隔离"跳/前扑能力"与"墙的干扰"——若 0 墙 flight/takeoff 起量,
#      说明梯度通、只是低墙仍有干扰; 若仍不起量, 问题在动作本身/reward
#   ② 其余配方不动 (POSTDIVE 0.40、entropy 0.002、stage4 降权跑步副本)
#   ③ warm-start 自 v3.9 收官 (run 2026-10-12_18-37-35 / model_62492)
#   ④ 后续课程: 0 → 0.5 → 1.5 → 3 → 5 → 12cm, 每档成功率打满再升
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.12
export DIVEWALL_POSTDIVE_PROB=0.40

LOAD_RUN=2026-10-12_18-37-35_microduck_roulade_divewall_run15
LOAD_CKPT=model_62492.pt
MAX_ITER=6000

LOG=logs/train_curric_stage4_v310.log
for ENVS in 2048 1024; do
  echo "=== [curric-stage4 v3.10] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER sink=$DIVEWALL_WALL_SINK (有效墙高 0cm) @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage4 v3.10 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage4 v3.10 结束 @ $(date) ==="

# 自动出 GIF (标准机位)
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif stage4 v3.10] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 8 --azimuth 100 --distance 1.6 \
  --video-width 1280 --video-height 720 --video-length 600 --num-envs 1 \
  > logs/gif_stage4_v310_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/curric_stage4_v310_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/curric_stage4_v310_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
