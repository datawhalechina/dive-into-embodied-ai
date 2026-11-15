#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Lean 课程 BL2: 起身链 (2026-10-27 课程收敛, 合并 B12+B14)
# 逻辑: BL1 全配置 + 起身链激活 (rise gate 150°/airborne 180°, 四层栈
#       12/8/4/4) + preroll_kick w2 填 20°-150° 真空 + stand_tax 3
#       (B13 弯路已知, 直接跳过; 只罚不教必退化)。
# 环境变量: B3+B7+B11+B12+B14 纯 flag 组合, 无新 env_cfg 版本块。
# warm: 默认接 BL1 产物 (主脚本自动传入)。
# 验收: frontier 冲过 ~130°, 最好已见过墙后站起
# ══════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.." || exit 1
export WANDB_MODE=offline
export DIVEWALL_STAGE=4
export DIVEWALL_WALL_SINK=0.100
export DIVEWALL_WALL_DIST=0.40,0.65
export DIVEWALL_POSTDIVE_PROB=0.0
export DIVEWALL_PHASE_B1=0
export DIVEWALL_PHASE_B3=1
export DIVEWALL_PHASE_B7=1
export DIVEWALL_PHASE_B11=1
export DIVEWALL_PHASE_B12=1
export DIVEWALL_PHASE_B14=1

LOAD_RUN="${LOAD_RUN:-}"
LOAD_CKPT="${LOAD_CKPT:-}"
MAX_ITER="${MAX_ITER:-6000}"
GIF_NAME="${GIF_NAME:-lean_bl2_rise}"

LOG=logs/train_leanBL2_wall.log
for ENVS in ${ENVS_LIST:-4096 2048}; do
  echo "=== [Lean BL2 rise chain] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  [ -n "$LOAD_RUN" ] || { echo "[error] 需 LOAD_RUN/LOAD_CKPT"; exit 1; }
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Lean BL2 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back (OOM guard)"
done
[ "$RC" -ne 0 ] && exit $RC
bash scripts/make_stage_gif.sh "$GIF_NAME"
