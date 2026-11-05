#!/bin/bash
# Stage-4 v2 (user 2026-10-09 17:35): post-dive spawn 反向课程。
# 25% envs 出生在"墙后 mid-dive 倒地姿态"（pitch 120-200°, tuck, 残余前向 ω），
# 给翻滚段喂 on-policy 数据（StandUp/Roulade 的 mid-roll 套路移植到墙流程）。
# warm-start 自 Stage-4 v1 收官 model_35498。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_POSTDIVE_PROB=0.25

LOAD_RUN=2026-10-09_12-02-49_microduck_roulade_divewall_run15
LOAD_CKPT=model_35498.pt
MAX_ITER=15000

for ENVS in 2048 1024; do
  echo "=== [curric-stage4 v2 postdive] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage4 v2 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage4 v2 结束 @ $(date) ==="
