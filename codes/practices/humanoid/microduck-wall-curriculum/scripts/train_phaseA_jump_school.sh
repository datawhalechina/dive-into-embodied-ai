#!/bin/bash
# Phase-A 跳跃学校 (user 2026-10-13 15:17 拍板: 先学会跳 → 再跳转前扑 → 后翻滚过渡):
#   复用 DIVEWALL_STAGE=2 跳远学校配置 (wall parked 4.5-5.5m, 只留跑步基底
#   + hop_forward + flop guard, 翻滚/墙侧项全关)
#   warm-start 自 v3.9 收官 (run 2026-10-12_18-37-35 / model_56493 一脉的
#   model_62492) —— 目标: 把"蹬地起跳-腾空-落地不摔"打牢, 再逐级接墙/前扑/翻滚
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2
export DIVEWALL_WALL_SINK=0.115
export DIVEWALL_POSTDIVE_PROB=0.0

LOAD_RUN=2026-10-12_18-37-35_microduck_roulade_divewall_run15
LOAD_CKPT=model_62492.pt
MAX_ITER=6000

LOG=logs/train_curric_phaseA_jump.log
for ENVS in 2048 1024; do
  echo "=== [Phase-A jump school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Phase-A jump school rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== Phase-A jump school 结束 @ $(date) ==="

# 自动出 GIF (标准机位)
RUN_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/2026-*_run15 | head -1)
CKPT=$(ls -t "$RUN_DIR"/model_*.pt | head -1)
echo "=== [gif Phase-A] ckpt=$CKPT @ $(date) ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/video_rollout.py \
  Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --camera trunk_base --elevation 8 --azimuth 100 --distance 1.6 \
  --video-width 1280 --video-height 720 --video-length 600 --num-envs 1 \
  > logs/gif_phaseA_jump_final.log 2>&1
MP4=$(ls -t "$RUN_DIR"/videos/play/*.mp4 2>/dev/null | head -1)
[ -n "$MP4" ] && ffmpeg -y -i "$MP4" -vf "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" logs/gifs/phaseA_jump_final.gif > /dev/null 2>&1 && echo "=== [gif done] logs/gifs/phaseA_jump_final.gif ==="
echo "=== ALL DONE @ $(date) ==="
