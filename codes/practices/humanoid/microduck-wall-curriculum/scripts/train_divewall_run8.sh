#!/bin/bash
# Dive-wall run8: pitch-gated takeoff (physics-driven fix).
# Run-7 result: 38% true flying leaps; failures cross low/clipped. Physics
# analysis (2026-10-02): standing COM z=0.146 vs wall top 0.12 -> only ~2cm
# margin. Clearing is a PITCH problem, not a jump problem: the trunk must be
# diving forward (|pitch|>=60°) at toe-off. Run-8 multiplies the airborne
# takeoff shaping by a |pitch| 50°→85° ramp (mdp.wall_takeoff_shaping), and
# bumps WALL_TAKEOFF_WEIGHT 4.0->6.0. Warm-start from run-7 model_75990.
# NOTE: GPU is co-tenant (algo-pi-agent training, ~2.5GB) -> 2048 envs primary.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-01_21-49-22_microduck_roulade_divewall_run7
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run7_model75990
if [ ! -f "$DST/model_75990.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_75990.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run7_model75990
LOAD_CKPT=model_75990.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run8 pitch-gated takeoff] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run8 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run8 结束 @ $(date) ==="
