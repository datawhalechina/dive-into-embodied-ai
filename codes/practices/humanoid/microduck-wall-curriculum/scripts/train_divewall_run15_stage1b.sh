#!/bin/bash
# Dive-wall run15 CURRICULUM stage 1: 学跑步 (user 2026-10-04).
# run14 GIF: duck STILL rolls before the wall — reward surgery lost.
# Curriculum: stage1 run-only (all dive/roll/pounce rewards OFF, wall at
# 1.6-2.2m for a real runway, premature-roll penalty stays ON so rolling
# remains the WORST option), stage2 +pounce, stage3 +past-wall roll.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=1

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-04_08-49-41_microduck_roulade_divewall_run14
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run14_model135493
if [ ! -f "$DST/model_135493.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_135493.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run14_model135493
LOAD_CKPT=model_135493.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [divewall-run15-stage1b RUN-ONLY anti-flop] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run15-stage1b rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run15-stage1b 结束 @ $(date) ==="
