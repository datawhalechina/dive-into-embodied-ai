#!/bin/bash
# 单队列顺序训练 14 任务，迭代减半
# 用法: nohup bash scripts/train_queue3.sh > logs/queue.log 2>&1 &

cd "$(dirname "$0")/.."
mkdir -p logs
export UV_HTTP_TIMEOUT=600

run() {  # run <task_id> <max_iterations>
  local TASK="$1" ITER="$2"
  local MARK="logs/queue_done_${TASK}.done"
  if [ -f "$MARK" ]; then echo "[skip] $TASK 已完成"; return; fi
  echo "=== [start] $TASK iter=$ITER @ $(date) ==="
  uv run train "$TASK" --env.scene.num-envs 4096 --agent.max_iterations "$ITER" \
    > "logs/train_${TASK}.log" 2>&1
  local RC=$?
  echo "=== [done rc=$RC] $TASK @ $(date) ==="
  [ $RC -eq 0 ] && touch "$MARK" || echo "[warn] $TASK 失败 rc=$RC，不标记完成"
}

# 按优先级排序：主任务 → 恢复类 → 花活
run Mjlab-Velocity-Flat-MicroDuck 25000
run Mjlab-Velocity-Rough-MicroDuck 25000
run Mjlab-Velocity-Flat-MicroDuck-Rollers 25000
run Mjlab-Velocity-Swizzle-MicroDuck 25000
run Mjlab-VelStand-Flat-MicroDuck 10000
run Mjlab-VelStand-Rough-MicroDuck 10000
run Mjlab-StandUp-Flat-MicroDuck 7500
run Mjlab-StandUp-Rough-MicroDuck 7500
run Mjlab-SitStand-Flat-MicroDuck 7500
run Mjlab-SitStand-Rough-MicroDuck 7500
run Mjlab-GroundPick-Flat-MicroDuck 10000
run Mjlab-GroundPick-Rough-MicroDuck 10000
run Mjlab-BallKick-Flat-MicroDuck 5000
run Mjlab-Roulade-Flat-MicroDuck 5000

echo "=== 队列全部结束 $(date) 完成数: $(ls logs/queue_done_Mjlab*.done 2>/dev/null | wc -l)/14 ==="
