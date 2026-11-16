#!/bin/bash
# L3 避障链：建缓存 → init(2D) → 加速训6k → eval_l3(>=50%)
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"
echo "=== [l3] 建缓存 $(date) ==="
if [ -f $NAS/cache_l3/info.json ]; then echo skip; else
  "$PY" build_cache_v5.py --out $NAS/cache_l3 --repo-id local/quaddle_l3 --dataset-root "$QD/outputs/dataset_l3/lerobot" > /root/l3_cache.log 2>&1
  grep -q DONE /root/l3_cache.log || { echo "!!! l3 cache fail"; exit 1; }
fi
echo "=== [l3] init $(date) ==="
"$PY" init_1d_v6.py --repo-id local/quaddle_l3 --dataset-root "$QD/outputs/dataset_l3/lerobot" --output-dir $NAS/init_l3 --steps 6000 > /root/l3_init.log 2>&1
grep -q INIT_OK /root/l3_init.log || { echo "!!! l3 init fail"; exit 1; }
echo "=== [l3] 加速训6k $(date) ==="
"$PY" train_cached_v5.py --resume $NAS/init_l3/checkpoints/last --repo-id local/quaddle_l3 \
    --dataset-root "$QD/outputs/dataset_l3/lerobot" --cache $NAS/cache_l3 --steps 6000 --decay 6000 \
    --batch 32 --save_freq 6000 --workers 4 --compile --output_dir $NAS/runs/l3 > /root/l3_run.log 2>&1
grep -q "training complete" /root/l3_run.log || { echo "!!! l3 train fail"; exit 1; }
echo "=== [l3] eval $(date) ==="
"$PY" eval_l3.py --ckpt $NAS/runs/l3/checkpoints/last/pretrained_model --out $NAS/runs/l3/rollout_l3.gif > /root/l3_eval.log 2>&1
grep -E 'L3_SUCCESS' /root/l3_eval.log
echo "=== [l3] 完成 $(date) ==="