#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# [已废弃 2026-10-28] BL2 (B12+B14 合并轮) 验收未达预期, 大人拍板回退原版 12 轮课程 (train_curriculum_scratch_to_5cm.sh)。本脚本与 BL1/BL2 阶段脚本留档不删。
# Lean 课程链 (9 阶段): 前传 → 5cm 墙, 每阶段自动出 GIF
# 2026-10-27 收敛: 12 → 9 轮 (B5+B7+B11→BL1, B12+B14→BL2; B19/B20 按大人指示保留分步)
# 预计 ~22h 等迭代 / ~13h 等样本量 (4096 envs, 4070 Ti)
# 原版 12 阶段备份: git tag curriculum-12stage-backup + train_curriculum_scratch_to_5cm.sh
#
# 用法:
#   ./train_curriculum_lean_to_5cm.sh [起始阶段]   # 默认 A3; 断点续跑传阶段名
#   AUTO=1  阶段间不停 (默认暂停等人工 GIF 验收)
#   ONLY=BL1 只跑一个阶段
# A3 warm 起点: 默认用 A3 脚本内置历史 ckpt (A2 毕业模型, 与原链同一起跑线)
# ══════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.." || exit 1
RUNS_DIR=logs/rsl_rl/microduck_roulade_dive

STAGES=(A3 A4 BL1 BL2 B18 B19 B20 B21 B24)
declare -A SCRIPT=(
  [A3]=train_phaseA3_jump_school.sh
  [A4]=train_phaseA4_jump_school.sh
  [BL1]=train_phaseBL1_wall_school.sh   # 2cm: B5+B7+B11 合并
  [BL2]=train_phaseBL2_wall_school.sh   # 起身链: B12+B14 合并
  [B18]=train_phaseB18_wall_school.sh   # midroll 反向课程
  [B19]=train_phaseB19_wall_school.sh   # 全链成立
  [B20]=train_phaseB20_wall_school.sh   # 陡扑
  [B21]=train_phaseB21_wall_school.sh   # 3.5cm
  [B24]=train_phaseB24_wall_school.sh   # 5cm (终点)
)
declare -A EXPECT=(
  [A3]="朝墙跑+真跳" [A4]="真跳确认(z≥0.14 抽帧)"
  [BL1]="8/8 干净过墙(落地躺平属预期)" [BL2]="frontier>130°, 争取过墙站起"
  [B18]="起身链贯通" [B19]="跑→跳扑→滚→跑 全链" [B20]="陡扑姿态稳定"
  [B21]="3.5cm 全链稳定" [B24]="5cm 全链稳定, reward 无回撤"
)

latest_run()  { ls -dt "$RUNS_DIR"/2026-* 2>/dev/null | head -1; }
latest_ckpt() { ls -t "$1"/model_*.pt 2>/dev/null | head -1; }

START=${1:-A3}
started=0
for ST in "${STAGES[@]}"; do
  [ "$ST" == "$START" ] && started=1
  [ "$started" -eq 0 ] && continue

  echo "════════ [$ST] $(date '+%F %T') ════════"
  unset $(env | grep -o '^DIVEWALL_[A-Z_0-9]*') 2>/dev/null
  unset LOAD_RUN LOAD_CKPT MAX_ITER

  # A3 用脚本内置历史 ckpt (原链同起点); 其余接上一阶段最新产物
  if [ "$ST" != "A3" ]; then
    RUN=$(latest_run); CKPT=$(latest_ckpt "$RUN")
    [ -z "$CKPT" ] && { echo "[error] 无可接 ckpt"; exit 1; }
    export LOAD_RUN="$(basename "$RUN")" LOAD_CKPT="$(basename "$CKPT")"
    echo "[warm] $LOAD_RUN / $LOAD_CKPT"
  fi

  bash "scripts/${SCRIPT[$ST]}" || { echo "[error] $ST 失败, 链条中止 @ $(date)"; exit 1; }

  echo "════════ [$ST 完成] GIF 验收: ${EXPECT[$ST]} ════════"
  if [ "$ONLY" == "$ST" ]; then echo "[info] ONLY=$ST 退出"; exit 0; fi
  if [ "${AUTO:-0}" != "1" ]; then
    read -p "验收通过? 回车继续, Ctrl-C 中止 > " _ || exit 0
  fi
done
echo "════════ Lean 全链完成: 5cm 毕业 @ $(date) ════════"
