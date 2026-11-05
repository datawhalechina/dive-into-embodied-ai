# Curriculum v3 Stage-1-run (user 2026-10-07): RUN SCHOOL v4 —
# prone_fall 罚 + lean 0-15° 平台 + 墙侧/死项 reward 全剔除.
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=1

LOAD_RUN=warmstart_curric21v2_stage1b
LOAD_CKPT=model_141492.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [curric-stage1-run v4] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage1-run rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage1-run 结束 @ $(date) ==="
