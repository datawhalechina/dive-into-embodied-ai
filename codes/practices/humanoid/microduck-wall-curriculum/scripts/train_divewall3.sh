#!/bin/bash
# Dive-wall run3: run-2 vaulted ONTO the wall (contact-cheap) instead of leaping.
# Changes: wall 6->10cm, FLAT any-contact penalty, one-shot flight-over-wall
# jackpot (+10), per-step cleared drip 2.0 -> 0.2. Fine-tune from run2
# model_57993 (copied into warmstart dir below).
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

SRC=logs/rsl_rl/microduck_roulade_dive/2026-09-30_23-59-47_microduck_roulade_divewall_run1
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run2_model57993
if [ ! -f "$DST/model_57993.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_57993.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run2_model57993
LOAD_CKPT=model_57993.pt
MAX_ITER=6000

for ENVS in 4096 2048; do
  echo "=== [divewall-run3] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run3 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== divewall-run3 结束 @ $(date) ==="
