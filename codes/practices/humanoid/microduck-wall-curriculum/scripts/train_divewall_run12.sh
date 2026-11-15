#!/bin/bash
# Dive-wall run12: immovable wall (釜底抽薪). Probe 2026-10-03 (run11
# model_111496, 300 steps): 0/16 real leaps — the policy sprints in from the
# run-10 runup spawn and BOWLS the 2kg wall 0.7-3.1 m, walking through the
# vacated plane (z_at_cross 0.073-0.119 < wall top 0.12; wall_up stayed 1.0
# so the tilt detector never fired and the -3.0/step shove penalty lost to
# runup+cleared reward). Fix: wall mass 2->500 kg, diaginertia 250x, floor
# friction 2.0 — physically un-shovable. Shove penalty / planted gating stay
# in the reward (now vestigial safety net). Warm-start run11 model_111496.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-02_23-54-50_microduck_roulade_divewall_run11
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run11_model111496
if [ ! -f "$DST/model_111496.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_111496.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run11_model111496
LOAD_CKPT=model_111496.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run12 immovable 500kg wall] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run12 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run12 结束 @ $(date) ==="
