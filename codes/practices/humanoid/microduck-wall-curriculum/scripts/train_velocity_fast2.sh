#!/bin/bash
# Velocity-fast run2: run1 (plain range widening) made the policy stand still
# at high commands — high-speed tracking paid nothing extra. Run2 adds:
#   - lin_vel_x range curriculum 0.5 -> 0.7 -> 0.9 -> 1.05 -> 1.2 (every 1500 iters)
#   - track_linear_velocity_highspeed bonus (gated |cmd|>=0.5), weight 0->1->1.5->2
# Warm-start from the STOCK velocity policy (not run1's degenerate one).
#
# 用法: nohup bash scripts/train_velocity_fast2.sh > logs/train_velocity_fast2.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=warmstart_velocity_model24999
LOAD_CKPT=model_24999.pt
MAX_ITER=12000   # 实测语义: 目标迭代数 = 载入迭代(24999) + 本值 => ~36999

for ENVS in 4096 2048; do
  echo "=== [velocity-fast2] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-VelocityFast2-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [velocity-fast2 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== velocity-fast2 结束 @ $(date) ==="
