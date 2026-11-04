#!/bin/bash
# Velocity-fast run3: run2 bug — curriculum stages used absolute
# common_step_counter thresholds; warm-start from model_24999 (counter ~600k)
# exceeded the max stage (144k) so range was 1.2 + bonus 2.0 from iter 1,
# i.e. no curriculum at all. Run3 rebases stage steps on the 25'000-iter
# warm-start baseline (see make_microduck_velocity_fast3_env_cfg).
# Warm-start from the STOCK velocity policy (same as run2).
#
# 用法: nohup bash scripts/train_velocity_fast3.sh > logs/train_velocity_fast3.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=warmstart_velocity_model24999
LOAD_CKPT=model_24999.pt
MAX_ITER=12000   # 实测语义: 目标迭代数 = 载入迭代(24999) + 本值 => ~36999

for ENVS in 4096 2048; do
  echo "=== [velocity-fast3] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-VelocityFast3-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [velocity-fast3 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== velocity-fast3 结束 @ $(date) ==="
