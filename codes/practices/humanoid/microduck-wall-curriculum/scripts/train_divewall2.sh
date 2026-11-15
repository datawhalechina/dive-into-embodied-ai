#!/bin/bash
# Dive-wall run2: run-1 escaped AROUND the 30cm-wide wall (side roll).
# Widened to 1.6m (0.8m lateral detour > anything reachable in a 1.5s
# episode). Fine-tune from divewall run-1 model_51994.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=warmstart_divewall_run1_model51994
LOAD_CKPT=model_51994.pt
MAX_ITER=6000

for ENVS in 4096 2048; do
  echo "=== [divewall-run2] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run2 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== divewall-run2 结束 @ $(date) ==="
