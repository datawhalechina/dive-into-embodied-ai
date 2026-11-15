#!/bin/bash
# Wait for the GPU to be free of algo-pi-agent jobs, then launch divewall run6.
cd "$(dirname "$0")/.." || exit 1

while true; do
  BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c "algo-pi-agent" || true)
  [ "$BUSY" -eq 0 ] && break
  sleep 180
done
echo "[watcher] GPU free of algo-pi-agent @ $(date)"

# small extra settle window in case another job is about to start
sleep 120
BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c "algo-pi-agent" || true)
if [ "$BUSY" -ne 0 ]; then echo "[watcher] GPU re-occupied, going back to wait"; exec bash "$0"; fi

export WANDB_MODE=offline
for ENVS in 4096 2048; do
  echo "=== [divewall-run6] envs=$ENVS @ $(date) ==="
  .venv/bin/python -m mjlab_microduck.train_cli Mjlab-RouladeDiveWall-Flat-MicroDuck \
    --env.scene.num-envs $ENVS --agent.max-iterations 8000 \
    --agent.resume True --agent.load-run warmstart_divewall_run3_actor --agent.load-checkpoint model_63992.pt
  RC=$?; echo "=== [divewall-run6 rc=$RC] @ $(date) ==="; [ $RC -eq 0 ] && break
done
echo "=== divewall-run6 结束 @ $(date) ==="
