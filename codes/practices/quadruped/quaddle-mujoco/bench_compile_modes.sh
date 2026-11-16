#!/bin/bash
# CUDA graphs 基准：对比 no-cudagraphs / max-autotune(含graphs) / reduce-overhead
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"
CACHE=/dataset/mtl_data/quaddle_vla/cache_emb_v51
for MODE in max-autotune-no-cudagraphs max-autotune reduce-overhead; do
  echo "=== MODE $MODE start $(date) ==="
  "$PY" train_cached_v5.py --cache "$CACHE" --no_resume_optim --steps 80 --batch 32 \
      --workers 4 --compile --compile-mode "$MODE" --log_every 10 --save_freq 1000 \
      --output_dir "/tmp/bench_$MODE" > "/root/bench_$MODE.log" 2>&1
  echo "--- $MODE steady ms/step + loss ---"
  grep -E 'ms/step' "/root/bench_$MODE.log" | tail -4
  echo "=== MODE $MODE end $(date) ==="
done
echo "BENCH_ALL_DONE"