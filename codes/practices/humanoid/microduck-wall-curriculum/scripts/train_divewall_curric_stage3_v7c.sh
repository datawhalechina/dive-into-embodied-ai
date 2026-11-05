#!/bin/bash
# Stage-3 v7c (user 2026-10-08): 前扑跳 + 空中旋转预习，warm-start 自 Stage-2 v7c。
# v7c 结构：走路配方降权版（track 2.0, vx 带 0.4-1.2 / pose 0.5 / upright 1.0 /
# air_time 1.5）+ hop_forward 6.0 (pitch ramp 8°→60°) + airborne_flop -2.0 (65°)
# + prone_fall -2.0 (50°, exempt_airborne=True — 腾空前扑不罚) + airborne_spin
# 2.0 (rate 封顶 3 rad/s)。entropy 0.002。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=3

LOAD_RUN=warmstart_stage2v7c_final
LOAD_CKPT=model_final.pt
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [curric-stage3 v7c dive-jump-school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.002 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [curric-stage3 v7c rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage3 v7c 结束 @ $(date) ==="
