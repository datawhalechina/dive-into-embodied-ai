#!/bin/bash
# Dive-roll run 3: continue from run2's model_23997 with more élan + heavier
# launch credit, targeting full_2pi (run-2 stalled at 41-49%, airborne_steps
# p50 = 0 — the roll reads as a grounded lunge, not a dive).
#   - élan 0.15-0.45 -> 0.35-0.7 m/s
#   - DIVE_LAUNCH_WEIGHT 2.0 -> 3.0
#   - vx gates widened to (0.3, 0.75) to match
#
# 用法: nohup bash scripts/train_dive_run3.sh > logs/train_dive_run3.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline

LOAD_RUN=2026-09-18_08-05-13_microduck_roulade_dive_run1
LOAD_CKPT=model_23997.pt
MAX_ITER=8000   # 实测语义: 目标迭代数 = 载入迭代(23997) + 本值 => ~31997

for ENVS in 4096 2048; do
  echo "=== [dive-run3] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDive-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [dive-run3 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 2048 (OOM guard)"
done
echo "=== dive-run3 结束 @ $(date) ==="
