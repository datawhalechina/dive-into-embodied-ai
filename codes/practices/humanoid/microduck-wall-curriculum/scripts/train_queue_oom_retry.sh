#!/bin/bash
# 重训 OOM 失败的 4 个任务，num-envs 降到 2048
# 用法: nohup bash scripts/train_queue_oom_retry.sh > logs/queue_retry.log 2>&1 &

cd "$(dirname "$0")/.."
mkdir -p logs
export UV_HTTP_TIMEOUT=600

run() {  # run <task_id> <max_iterations>
  local TASK="$1" ITER="$2"
  local MARK="logs/queue_done_${TASK}.done"
  if [ -f "$MARK" ]; then echo "[skip] $TASK 已完成"; return; fi
  echo "=== [start] $TASK iter=$ITER envs=2048 @ $(date) ==="
  uv run train "$TASK" --env.scene.num-envs 2048 --agent.max_iterations "$ITER" \
    > "logs/train_${TASK}.log" 2>&1
  local RC=$?
  echo "=== [done rc=$RC] $TASK @ $(date) ==="
  [ $RC -eq 0 ] && touch "$MARK" || echo "[warn] $TASK 失败 rc=$RC，不标记完成"
}

run Mjlab-Velocity-Rough-MicroDuck 25000
run Mjlab-VelStand-Rough-MicroDuck 10000
run Mjlab-SitStand-Flat-MicroDuck 7500
run Mjlab-SitStand-Rough-MicroDuck 7500

echo "=== 重训队列结束 $(date) ==="
