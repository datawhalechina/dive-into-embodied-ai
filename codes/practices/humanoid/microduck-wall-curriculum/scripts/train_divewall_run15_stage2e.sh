#!/bin/bash
# Dive-wall run15 stage 2e: WALL-HEIGHT CURRICULUM (2026-10-05).
# stage2d post-mortem: takeoff = 0 for 5k+ iters — push (grounded) and
# takeoff (airborne) are per-step mutually exclusive; the duck rationally
# farms grounded push and never leaves the ground. Curriculum: sink the
# wall 5cm (effective 7cm) so existing hops can clear it BY ACCIDENT →
# flight jackpot (15.0) gets sampled → real gradient. Raise later.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2
export DIVEWALL_WALL_SINK=0.05

LOAD_RUN=warmstart_divewall_run15s2d_model156750
LOAD_CKPT=model_156750.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [divewall-run15-stage2e LOW-WALL-7cm] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run15-stage2e rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run15-stage2e 结束 @ $(date) ==="
