#!/bin/bash
# Stage-1-run v7 (user 2026-10-08): WALK-RECIPE port — 走路配方直接搬来教跑步。
# v4-v6 的 wall_runup 门控 shaping 从静止起步梯度≈0（vx_lo 0.25 门以下零分）；
# velocity env 教会鸭子走路的配方是：dense gaussian 速度跟踪 + pose/upright
# 姿态锚 + air_time。本版 Stage-1 reward:
#   track_linear_velocity 2.0 | variable_posture 1.0 | upright 2.0 |
#   feet_air_time 3.0 | prone_fall -2.0（v6 地板保留）
# 指令固定 forward-only vx∈[0.2,0.6], vy=0, ωz=0；actor obs 不变（61 维，
# 指令槽仍是 zero-padding——指令是常量带，policy 不需要看见它）。
# 其余 v6 设定保留：from scratch、初速 0、spawn 全 standing、墙 4.5-5.5m、
# prone_cut z<0.07 截断、无 DR、action_rate 固定 -0.01。
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=1

LOAD_RUN=""
LOAD_CKPT=""
MAX_ITER=6000

for ENVS in 2048 1024; do
  echo "=== [curric-stage1-run v7 walk-recipe] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    ${LOAD_RUN:+--agent.resume True --agent.load-run "$LOAD_RUN" --agent.load-checkpoint "$LOAD_CKPT"}
  RC=$?
  echo "=== [curric-stage1-run v7 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back to 1024 (OOM guard)"
done
echo "=== curric-stage1-run v7 结束 @ $(date) ==="
