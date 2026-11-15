#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Lean 课程 BL1: 2cm 墙学校 (2026-10-27 课程收敛, 合并 B5+B7+B11)
# 逻辑: 三轮的"已知好配置"一步到位 —— 出生 0.40-0.65m (B5) +
#       air_cross_gate 防趴蹭 (B7) + takeoff vx/vz 双硬门 (B11)。
# 墙高取 2cm (SINK 0.100): 起身链阶段的历史工作档位。
# 环境变量: 纯 flag 组合 (B3+B7+B11), 无新 env_cfg 版本块。
# warm: 默认接 A4 产物 (课程链主脚本自动传入最新 run)。
# 验收: 8/8 干净过墙 (落地躺平属预期, 起身在 BL2 解决)
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

LOAD_RUN="${LOAD_RUN:-}"   # 课程链主脚本自动传入; 单跑时可手动指定
LOAD_CKPT="${LOAD_CKPT:-}"
MAX_ITER="${MAX_ITER:-6000}"
GIF_NAME="${GIF_NAME:-lean_bl1_2cm}"

LOG=logs/train_leanBL1_wall.log
for ENVS in ${ENVS_LIST:-4096 2048}; do  # perf: 4096 档 (+31%), OOM 回落 2048
  echo "=== [Lean BL1 2cm wall school] warm=$LOAD_RUN/$LOAD_CKPT envs=$ENVS max_iter=$MAX_ITER @ $(date) ==="
  [ -n "$LOAD_RUN" ] || { echo "[error] 需 LOAD_RUN/LOAD_CKPT (经主脚本或手动)"; exit 1; }
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$MAX_ITER" \
    --agent.algorithm.entropy-coef 0.004 \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$LOAD_CKPT"
  RC=$?
  echo "=== [Lean BL1 rc=$RC] envs=$ENVS @ $(date) ==="
  [ "$RC" -eq 0 ] && break
  echo "[warn] rc=$RC at envs=$ENVS — falling back (OOM guard)"
done
[ "$RC" -ne 0 ] && exit $RC
bash scripts/make_stage_gif.sh "$GIF_NAME"
