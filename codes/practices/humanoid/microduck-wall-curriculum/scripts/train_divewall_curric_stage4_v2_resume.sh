#!/bin/bash
# Stage-4 v2 续训：从昨晚中断的 run 2026-10-09_17-39-30 (model_47500) 恢复，跑完剩余 ~3000 iter。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_POSTDIVE_PROB=0.25

LOAD_RUN=2026-10-09_17-39-30_microduck_roulade_divewall_run15
LOAD_CKPT=model_47500.pt
MAX_ITER=3000

for ENVS in 2048 1024; do
  echo "=== [curric-stage4 v2 resume] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage4 v2 resume rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage4 v2 resume 结束 @ $(date) ==="
