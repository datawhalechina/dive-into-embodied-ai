#!/bin/bash
# Dive-wall run15 stage 2b: fix 扑得太早 (user 2026-10-05).
# stage2 GIF: duck pounces BEFORE the wall. Root cause: premature_roll
# penalty only fires GROUNDED; early airborne dives were free. Fix: new
# wall_premature_dive penalty (airborne+pitch>15° / grounded pitch>32°
# beyond 30cm, -2.0/step) + premature_roll zone aligned 0.15->0.30m.
# Still DIVEWALL_STAGE=2 (pounce chain not yet earned; roll stays off).
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-05_09-15-52_microduck_roulade_divewall_run15
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run15s2_model149492
LATEST=$(ls -t "$SRC"/model_*.pt | head -1 | xargs basename)
NUM=${LATEST#model_}; NUM=${NUM%.pt}
if [ ! -f "$DST/$LATEST" ]; then
  mkdir -p "$DST" && cp "$SRC/$LATEST" "$DST/" || exit 1
fi

LOAD_RUN=warmstart_divewall_run15s2_model$NUM
LOAD_CKPT=$LATEST
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run15-stage2b ANTI-EARLY-DIVE] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run15-stage2b rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run15-stage2b 结束 @ $(date) ==="
