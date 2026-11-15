#!/bin/bash
# Dive-wall run13 (plan A: jump-impulse rewards). Run-12 GIF 2026-10-03:
# duck ROLLS UP to the wall then FREEZES planted in front of it — approach
# is solved (runup carried over), missing is the commit to jump. Takeoff
# reward ate only ~0.003 over 8k iters (pitch gate 50-85° + airborne + vx
# never simultaneously true), so the policy never tasted it. Fix:
# (1) WALL_PUSH_WEIGHT 1.5 -> 6.0: pay grounded vz>0 in front of the wall
#     (wall_push_shaping needs no pitch/air/vx — any hop impulse pays);
# (2) WALL_TAKEOFF_PITCH (50,85) -> (30,70): partial forward-lean counts.
# 500kg immovable wall (run-12) and everything else unchanged.
# Warm-start run12 model_119495.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

SRC=logs/rsl_rl/microduck_roulade_dive/2026-10-03_08-59-36_microduck_roulade_divewall_run12
DST=logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run12_model119495
if [ ! -f "$DST/model_119495.pt" ]; then
  mkdir -p "$DST" && cp "$SRC"/model_119495.pt "$DST"/ || exit 1
fi

LOAD_RUN=warmstart_divewall_run12_model119495
LOAD_CKPT=model_119495.pt
MAX_ITER=8000

for ENVS in 2048 1024; do
  echo "=== [divewall-run13 jump-impulse: push 1.5->6.0, takeoff pitch 50/85->30/70] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run13 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== divewall-run13 结束 @ $(date) ==="
