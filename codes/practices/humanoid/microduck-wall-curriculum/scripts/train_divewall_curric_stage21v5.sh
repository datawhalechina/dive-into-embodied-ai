# Curriculum v2 stage 21-rev5: soft vx gate + bounded action_rate (from v4 golden ckpt 142000)
# (v2 verdict 2026-10-06: duck stood still — 0.3 free floor paid stationary
#  micro-hops; v4: no forward speed -> no hop pay). Clean stage1b lineage.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=21

LOAD_RUN=warmstart_curric21v5_golden
LOAD_CKPT=model_142000.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [curric-stage21v5 jump-school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage21v5 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage21v5 结束 @ $(date) ==="
