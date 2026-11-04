#!/bin/bash
# Stage-1-run v7b (user 2026-10-08 13:35): v7 walk-recipe 的 std 爆炸修复重启。
# v7 病理：reward ~iter170 见顶 17.45 后 action std 从 1.0 单调爆到 77+
# （reward 饱和后 entropy_coef 0.01 成主导梯度）。v7b：
#   ① warm-start 自 v7 峰值 checkpoint model_250；
#   ② entropy_coef 0.01 → 0.002。
# Reward / 环境与 v7 完全一致（走路配方：tracking 2.0 + pose 1.0 + upright
# 2.0 + air_time 3.0 + prone_fall -2.0，forward-only 指令）。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=1

LOAD_RUN=warmstart_stage1v7_peak_model250
LOAD_CKPT=model_250.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [curric-stage1-run v7b walk-recipe, entropy 0.002] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage1-run v7b rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage1-run v7b 结束 @ $(date) ==="
