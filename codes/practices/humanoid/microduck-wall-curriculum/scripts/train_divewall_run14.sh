#!/bin/bash
# Dive-wall run14 (plan A': bind jump to dive+run-up). Run-13 GIF 2026-10-04:
# duck has NO run-up (runup stalled 0.07 all 8k iters) and rolls into the
# wall from a standstill — push 0.16 farmed via tiny STATIONARY hops (vz-only
# credit), flight stuck at 0.021. Fix:
# (1) wall_push_shaping gains a forward-pitch ramp (10-30°, like takeoff) and
#     a vx factor x(0.3+0.7*clamp01(vx/1.0)) — a stationary hop now pays <=30%
#     of the term; jump and dive must co-occur to earn full credit;
# (2) WALL_RUNUP_WEIGHT 2.0 -> 4.0: double the pull toward leaning run-up
#     speed in the runway (user wants a real run before the dive).
# 500kg wall, spawn 0.85-1.2m, everything else inherited from run-13.
# Warm-start run13 model_127494.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-03_21-15-44_microduck_roulade_divewall_run13
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run13_model127494
if [ ! -f "$DST/model_127494.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_127494.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run13_model127494
LOAD_CKPT=model_127494.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run14b bind jump+dive + past-wall roll gate: push pitch-gate 10/30 + vx factor, runup x2] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run14 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run14 结束 @ $(date) ==="
