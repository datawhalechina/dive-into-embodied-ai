#!/bin/bash
# Velocity-fast run1: widen lin_vel_x (-0.4,0.4) -> (-0.4,1.2) so the walk
# can reach the dive élan range (0.5-1.2 m/s). Warm-start from the stock
# velocity policy model_24999 (25k iters, flat). No reward changes.
#
# 用法: nohup bash scripts/train_velocity_fast.sh > logs/train_velocity_fast.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=warmstart_velocity_model24999
LOAD_CKPT=model_24999.pt
MAX_ITER=10000   # 实测语义: 目标迭代数 = 载入迭代(24999) + 本值 => ~34999

for ENVS in 4096 2048; do
  echo "=== [velocity-fast] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-VelocityFast-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [velocity-fast rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== velocity-fast 结束 @ $(date) ==="
