#!/bin/bash
# Dive-wall run15 CURRICULUM stage 2: 带初速前扑 (user 2026-10-05).
# stage1b passed user check (run-only, anti-flop). Warm-start stage1b
# model_141492, DIVEWALL_STAGE=2 => runup + push/takeoff/flight/cleared ON,
# roll rewards stay OFF until stage3.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-04_15-04-22_microduck_roulade_divewall_run15
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run15s1b_model141492
if [ ! -f "$DST/model_141492.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_141492.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run15s1b_model141492
LOAD_CKPT=model_141492.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run15-stage2 POUNCE] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run15-stage2 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run15-stage2 结束 @ $(date) ==="
