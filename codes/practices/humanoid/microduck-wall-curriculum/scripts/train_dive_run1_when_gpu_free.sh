#!/bin/bash
# 等 GPU 空闲(其他项目任务结束后)再启动 dive-roll run1。
# 每 60s 查一次空闲显存,>= NEED_MB 才启动;最多等 MAX_WAIT_MIN 分钟。
cd "$(dirname "$0")/.." || exit 1

NEED_MB=8000
MAX_WAIT_MIN=360
waited=0
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -ge "$NEED_MB" ]; then
    echo "=== GPU 空闲 ${FREE}MiB >= ${NEED_MB}MiB,启动训练 @ $(date) ==="
    exec bash scripts/train_dive_run1.sh
  fi
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    echo "=== 等待超时 ${MAX_WAIT_MIN}min,空闲仅 ${FREE}MiB,放弃启动 @ $(date) ==="
    exit 2
  fi
  echo "[wait] free=${FREE}MiB < ${NEED_MB}MiB,已等 ${waited}min @ $(date)"
  sleep 60
  waited=$((waited + 1))
done
