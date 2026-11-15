#!/bin/bash
# Dive-roll run 4: continue from run3's model_31996 with real-flight launch
# credit (height gain above STAND_Z + airborne duration in the <90° window)
# and élan widened to cover the 1.2 m/s trigger regime in-distribution.
#   - élan 0.35-0.7 -> 0.5-1.2 m/s
#   - DIVE_LAUNCH_WEIGHT 3.0 -> 6.0, func -> roulade_dive_launch_v2
#     (vx gate 0.4-1.25, pays airborne steps + z-above-stand_z gain)
#
# 用法: nohup bash scripts/train_dive_run4.sh > logs/train_dive_run4.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=2026-09-27_21-31-04_microduck_roulade_dive_run1
LOAD_CKPT=model_31996.pt
MAX_ITER=10000   # 实测语义: 目标迭代数 = 载入迭代(31996) + 本值 => ~41996

for ENVS in 4096 2048; do
  echo "=== [dive-run4] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDive-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [dive-run4 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== dive-run4 结束 @ $(date) ==="
