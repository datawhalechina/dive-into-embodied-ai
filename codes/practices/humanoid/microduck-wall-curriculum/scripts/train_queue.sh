#!/bin/bash
# 顺序训练 14 个 MicroDuck 任务（4096 envs，默认迭代数）
# 用法: nohup bash train_queue.sh > logs/queue.log 2>&1 &

cd "$(dirname "$0")/.."
mkdir -p logs
export UV_HTTP_TIMEOUT=600

TASKS=(
  "Mjlab-Velocity-Flat-MicroDuck"
  "Mjlab-Velocity-Rough-MicroDuck"
  "Mjlab-VelStand-Flat-MicroDuck"
  "Mjlab-VelStand-Rough-MicroDuck"
  "Mjlab-StandUp-Flat-MicroDuck"
  "Mjlab-StandUp-Rough-MicroDuck"
  "Mjlab-SitStand-Flat-MicroDuck"
  "Mjlab-SitStand-Rough-MicroDuck"
  "Mjlab-GroundPick-Flat-MicroDuck"
  "Mjlab-GroundPick-Rough-MicroDuck"
  "Mjlab-BallKick-Flat-MicroDuck"
  "Mjlab-Roulade-Flat-MicroDuck"
  "Mjlab-Velocity-Flat-MicroDuck-Rollers"
  "Mjlab-Velocity-Swizzle-MicroDuck"
)

QUEUE_FILE=logs/queue_done.txt
touch "$QUEUE_FILE"

echo "=== 训练队列启动 $(date) ==="

for TASK in "${TASKS[@]}"; do
  if grep -qxF "$TASK" "$QUEUE_FILE"; then
    echo "[skip] $TASK 已完成"
    continue
  fi
  echo "=== [start] $TASK @ $(date) ==="
  uv run train "$TASK" --env.scene.num-envs 4096 \
    > "logs/train_${TASK}.log" 2>&1
  RC=$?
  echo "=== [done rc=$RC] $TASK @ $(date) ==="
  if [ $RC -eq 0 ]; then
    echo "$TASK" >> "$QUEUE_FILE"
  else
    echo "[warn] $TASK 退出码 $RC，已跳过（不标记完成）"
  fi
done

echo "=== 队列结束 $(date) 共完成 $(wc -l < "$QUEUE_FILE")/14 ==="
