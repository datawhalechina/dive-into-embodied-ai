#!/bin/bash
# Dive-wall run1: obstacle-forced dive. Run-4's dive reads as "pushed over"
# not "leaping". A heavy 6cm hurdle is placed 0.22–0.30 m ahead of every
# (standing) spawn: collision penalty + cleared-plane bonus make an active
# VAULT the only route to a completed roll. Wall not in obs (scaffolding
# only, proprioceptive policy unchanged). Warm-start from dive run-4
# (model_41995: 100% rise, 364.8° roll, real flight).
#
# 用法: nohup bash scripts/train_divewall.sh > logs/train_divewall_run1.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=warmstart_dive_run4_model41995
LOAD_CKPT=model_41995.pt
MAX_ITER=10000

for ENVS in 4096 2048; do
  echo "=== [divewall-run1] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run1 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== divewall-run1 结束 @ $(date) ==="
