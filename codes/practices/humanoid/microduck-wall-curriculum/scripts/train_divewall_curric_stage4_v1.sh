#!/bin/bash
# Stage-4 v1 (user 2026-10-09): 全流程 跑→前扑→过墙→翻滚。
# DIVEWALL_STAGE=4 (= unset full config)：墙回到真实距离，roll/dive/wall 全量
# 奖励开启。warm-start 自 Stage-3 v7e 收官（hop 0.94, rc=0）。
# spin 探索铺路（Kimi爪建议）：初始角速度随机化 + entropy 0.002 保持。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4

LOAD_RUN=2026-10-08_22-24-39_microduck_roulade_divewall_run15
LOAD_CKPT=model_20499.pt
MAX_ITER=15000

for ENVS in 2048 1024; do
  echo "=== [curric-stage4 v1 full-pipeline] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage4 v1 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage4 v1 结束 @ $(date) ==="
