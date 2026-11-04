#!/bin/bash
# Wait for divewall run3 to finish, then build actor-only warmstart and launch run4.
cd "$(dirname "$0")/.." || exit 1
LOG=logs/train_divewall_run3.log

# 1. wait for run3 completion marker
while ! grep -q "divewall-run3 结束" "$LOG"; do sleep 120; done
echo "[watcher] run3 done @ $(date)"

# 2. locate run3 final checkpoint (newest divewall_run? dir with model_6*.pt)
RUN3_DIR=$(ls -dt logs/rsl_rl/microduck_roulade_dive/*divewall_run3* 2>/dev/null | head -1)
[ -z "$RUN3_DIR" ] && { echo "[watcher] no run3 dir found"; exit 1; }
CKPT=$(ls -t "$RUN3_DIR"/model_*.pt | head -1)
ITER=$(basename "$CKPT" .pt | sed 's/model_//')
echo "[watcher] warmstart from $CKPT (iter $ITER)"

# 3. build actor-only warmstart (critic obs grew +3)
.venv/bin/python scripts/make_actor_warmstart.py \
  --src "$CKPT" \
  --out logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run3_actor \
  --iter "$ITER" || exit 1

# 4. launch run4
export WANDB_MODE=offline
LOAD_RUN=warmstart_divewall_run3_actor
LOAD_CKPT="model_${ITER}.pt"
MAX_ITER=8000

for ENVS in 4096 2048; do
  echo "=== [divewall-run4] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [divewall-run4 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
done
echo "=== divewall-run4 结束 @ $(date) ==="
