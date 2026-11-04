#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# 课程链主脚本：从零(前传 ckpt)到越过 5cm 墙 一键接力
# 2026-10-27 整理。配套 perf 配置: 4096 envs (1.5x, 见
# docs/perf/2026-10-26-training-speed-optimization.md)
#
# 用法:
#   ./train_curriculum_scratch_to_5cm.sh [起始阶段]
#     不带参数 = 从 A3 开始全链
#     ./train_curriculum_scratch_to_5cm.sh B18   = 从 B18 开始续跑
#   环境变量:
#     BOOTSTRAP_RUN / BOOTSTRAP_CKPT  A3 的 warm 起点 (前传平地翻滚 ckpt,
#                                     必填, 除非 logs 里已有历史 run 可接)
#     AUTO=1     跳过阶段间人工确认 (默认每阶段结束暂停等 GIF 验收)
#     ONLY=A3    只跑指定阶段后退出
#
# 设计原则 (见 Obsidian 迭代史「升档心法」):
#   ① 每阶段结束自动暂停等人工 GIF 验收 —— 不做早停, 不自动升档
#   ② warm 来源 = 上一阶段刚产出的最新 run 最新 ckpt (自动探测)
#   ③ 每阶段前 unset 全部 DIVEWALL_*, 防止上阶段环境变量泄漏
#      (如 B21=1 泄漏进 B5 会多注册 reward 项)
# ══════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/.." || exit 1
RUNS_DIR=logs/rsl_rl/microduck_roulade_dive

# 最小成功路线 12 轮 (A5/B12b 巩固轮按验收结果手动插入)
# v2 谱系 (2026-10-31 对齐实际成功路线): B18 之后走 v2 脚本, 终点 = B22v2 5cm。
# ⚠ B18 注意: 实际成功的 B18 是 git worktree @00cd9b6 (原版代码: nan guard
# 默认开 + njmax 启发式) 跑出来的; 在 HEAD 代码上跑 B18 需确认默认未漂移
# (见迭代史第七幕五连败教训)。
STAGES=(A3 A4 B5 B7 B11 B12 B14 B18 B19v2 B20v2 B21v2 B22v2)
declare -A SCRIPT=(
  [A3]=train_phaseA3_jump_school.sh   # 跳学校: hop 15/track 1/航向锚定
  [A4]=train_phaseA4_jump_school.sh   # 跳学校毕业: 落地结算+高度门+超速罚
  [B5]=train_phaseB5_wall_school.sh   # 1cm 临界墙: 出生拉近 0.40-0.65m
  [B7]=train_phaseB7_wall_school.sh   # 2cm 真降墙(sink bug 修复后)+air_cross_gate
  [B11]=train_phaseB11_wall_school.sh # takeoff vx/vz 硬门+起身栈 gate
  [B12]=train_phaseB12_wall_school.sh # 起身链激活: rise gate 150°/180°
  [B14]=train_phaseB14_wall_school.sh # preroll_kick, warm 回接 B12 能力已证 ckpt
  [B18]=train_phaseB18_wall_school.sh # 反向课程: midroll 0.4 + overhead_drive
  [B19v2]=train_phaseB19v2_wall_school.sh # 全链成立: midroll 0.22 + flight_pitch + NJMAX_NONE/B18B=0 修正
  [B20v2]=train_phaseB20v2_wall_school.sh # 陡扑门 20/65° + takeoff 权重×2 (v2 对齐 worktree 物理)
  [B21v2]=train_phaseB21v2_wall_school.sh # 3.5cm 墙 (SINK 0.085), reward 零改动
  [B22v2]=train_phaseB22v2_5cm_wall.sh # 5cm 终点 (SINK 0.070) ← 最终目标 (2026-10-31 收官, reward 79.4)
)
declare -A EXPECT=(  # 验收要点 (GIF)
  [A3]="朝墙跑+真跳"
  [A4]="真跳确认(抽帧躯干 z≥0.14)"
  [B5]="cleared/flight 首次非零"
  [B7]="8/8 干净过墙"
  [B11]="干净过墙(落地躺平属预期, 起身在 B12 解决)"
  [B12]="首次过墙后站起来"
  [B14]="frontier 冲过 ~130°"
  [B18]="起身链贯通"
  [B19v2]="跑→跳扑→滚→跑 全链"
  [B20v2]="陡扑姿态稳定"
  [B21v2]="3.5cm 全链稳定"
  [B22v2]="5cm 全链稳定 (已验收: reward 79.4 新高)"
)

latest_run()  { ls -dt "$RUNS_DIR"/2026-* 2>/dev/null | head -1; }
latest_ckpt() { ls -t "$1"/model_*.pt 2>/dev/null | head -1; }

START=${1:-A3}
started=0
for ST in "${STAGES[@]}"; do
  [ "$ST" == "$START" ] && started=1
  [ "$started" -eq 0 ] && continue

  echo "════════ 阶段 $ST ($(basename "${SCRIPT[$ST]}")) ════════"

  # ③ 清理上阶段环境变量泄漏
  unset $(env | grep -o '^DIVEWALL_[A-Z_0-9]*') 2>/dev/null
  unset LOAD_RUN LOAD_CKPT MAX_ITER

  # ② warm 来源: A3 用 BOOTSTRAP 或最新历史 run; 其余用上一阶段产物
  if [ "$ST" == "A3" ] && [ -n "$BOOTSTRAP_RUN" ]; then
    export LOAD_RUN="$BOOTSTRAP_RUN" LOAD_CKPT="$BOOTSTRAP_CKPT"
  else
    RUN=$(latest_run)
    CKPT=$(latest_ckpt "$RUN")
    if [ -z "$CKPT" ]; then
      echo "[error] $RUNS_DIR 下无可用 run/ckpt — 请设 BOOTSTRAP_RUN/BOOTSTRAP_CKPT"; exit 1
    fi
    export LOAD_RUN="$(basename "$RUN")" LOAD_CKPT="$(basename "$CKPT")"
  fi
  echo "[warm] $LOAD_RUN / $LOAD_CKPT"

  bash "scripts/${SCRIPT[$ST]}" || { echo "[error] 阶段 $ST 失败, 链条中止"; exit 1; }

  echo "════════ 阶段 $ST 完成 — 请 GIF 验收: ${EXPECT[$ST]} ════════"
  echo "        (出 GIF 记得带该阶段全套 DIVEWALL_* 环境变量, 见阶段脚本头)"
  if [ "$ONLY" == "$ST" ]; then echo "[info] ONLY=$ST, 退出"; exit 0; fi
  if [ "${AUTO:-0}" != "1" ]; then
    read -p "验收通过? 回车继续下一阶段, Ctrl-C 中止 > " _ || exit 0
  fi
done
echo "════════ 全链完成: 5cm 墙课程毕业 ════════"
