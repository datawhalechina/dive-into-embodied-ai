#!/bin/bash
# Stage-2-jump v7c (user 2026-10-08): 跳远学校，warm-start 自 Stage-1 v7b。
# v7c 结构：继承走路配方降权版（track 2.0, vx 带 0.3-0.9 / pose 0.5 /
# upright 1.0 / air_time 1.5）+ hop_forward 6.0 (soft vx gate) + airborne_flop
# -2.0 + prone_fall -2.0 (40°)。entropy 0.002（v7 的 std 爆炸教训）。
# 用法: 先把 Stage-1 v7b 的最终 checkpoint 拷到 warmstart_stage1v7b_final/
# 再 nohup bash scripts/train_divewall_curric_stage2_v7c.sh > logs/train_curric_stage2_v7c.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=2

LOAD_RUN=warmstart_stage1v7b_final
LOAD_CKPT=model_final.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [curric-stage2 v7c jump-school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage2 v7c rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage2 v7c 结束 @ $(date) ==="
