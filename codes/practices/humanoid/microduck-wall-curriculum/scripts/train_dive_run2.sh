#!/bin/bash
# Dive-roll run 2: continue from run1's model_15998 with the densified reward
# (roulade_dive_approach 连续 shaping, run-1 的乘积门 6000 轮只有 ~0.0005,
# 稀疏不可发现) + 提高的前向初速 (0.15–0.45 m/s)。
#
# 用法: nohup bash scripts/train_dive_run2.sh > logs/train_dive_run2.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=2026-09-18_00-58-24_microduck_roulade_dive_run1
LOAD_CKPT=model_15998.pt
MAX_ITER=8000   # 实测语义: 目标迭代数 = 载入迭代(15998) + 本值 => ~23998

for ENVS in 4096 2048; do
  echo "=== [dive-run2] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDive-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [dive-run2 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== dive-run2 结束 @ $(date) ==="
