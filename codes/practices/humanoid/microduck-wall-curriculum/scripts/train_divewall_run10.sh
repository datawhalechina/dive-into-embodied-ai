#!/bin/bash
# Dive-wall run8c: fix run8b (tilt-detector yaw bug + 9cm wall bug). anti-shove+tiltfix+12cm-wall patch on top of run8's pitch-gated takeoff.
# Probe finding (2026-10-02): 8/12 "crossed" envs crossed at z=0.046-0.065 —
# the duck SHOVED the 2kg free-joint wall aside and walked through the
# vacated plane. Fix: (1) wall_shove_penalty, flat per-step -3.0 while the
# wall is displaced >1.5cm or tilted >15°; (2) cleared/flight bonuses only
# pay when the wall is still planted. Pitch-gated takeoff (run8) retained.
# Warm-start from run8 model_76750 (pitch-gate already ~1 iter in).
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-02_14-04-24_microduck_roulade_divewall_run9
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run9_model95498
if [ ! -f "$DST/model_95498.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_95498.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run9_model95498
LOAD_CKPT=model_95498.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run10 run-up decomposition: spawn 0.85-1.2m + leaning run-up reward] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run10 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run10 结束 @ $(date) ==="
