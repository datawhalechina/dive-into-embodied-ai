#!/bin/bash
# Run-6 kip-up CONTINUE: r09 (500-iter tuning probe, v6 rewards) -> full-scale
# consolidation. Goal: kip-up "pop" — airborne credit + rise_velocity + stand_tax
# push an explosive rise instead of run-5's slow crunch / frozen-fold parking.
#
# Resume fix (09-14 crash): load-run and load-checkpoint must be passed
# SEPARATELY — a full path in load-checkpoint collides with the auto-picked
# latest load-run and get_checkpoint_path fails before iteration 0.
#
# 用法: nohup bash scripts/train_run6_continue.sh > logs/train_run6_continue.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=2026-09-16_00-44-03_microduck_roulade_run6_kipup
LOAD_CKPT=model_7250.pt
# max_iterations = iterations TO RUN THIS SESSION. 7250 done -> 9998 target,
# so ~2750 to go (was 4500 from r09's 5498 before the first leg was killed).
MAX_ITER=2750

for ENVS in 4096 2048; do
  echo "=== [run6-continue] resume=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-Roulade-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [run6-continue rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== run6-continue 结束 @ $(date) ==="
