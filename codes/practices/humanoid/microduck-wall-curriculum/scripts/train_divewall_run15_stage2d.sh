#!/bin/bash
# Dive-wall run15 stage 2d: rebalanced anti-early-dive (user 2026-10-05).

# then oscillated into total collapse — premature_dive -2.0 too strong vs

# checkpoint model_151500 (iter 150750-151500, dive ~-0.03).
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-05_12-06-33_microduck_roulade_divewall_run15
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run15s2b_model157000
if [ ! -f "$DST/model_151500.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_151500.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run15s2b_model157000
LOAD_CKPT=model_151500.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run15-stage2d TAKEOFF-RAMP] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run15-stage2d rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run15-stage2d 结束 @ $(date) ==="
