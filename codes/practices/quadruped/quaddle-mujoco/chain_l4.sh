#!/bin/bash
# L4 条件组合链：建缓存 → init(2D) → 加速训6k → eval_l4
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"
echo "=== [l4] 建缓存 $(date) ==="
if [ -f $NAS/cache_l4/info.json ]; then echo skip; else
  "$PY" build_cache_v5.py --out $NAS/cache_l4 --repo-id local/quaddle_l4 --dataset-root "$QD/outputs/dataset_l4/lerobot" > /root/l4_cache.log 2>&1
  grep -q DONE /root/l4_cache.log || { echo "!!! l4 cache fail"; exit 1; }
fi
echo "=== [l4] init $(date) ==="
"$PY" init_1d_v6.py --repo-id local/quaddle_l4 --dataset-root "$QD/outputs/dataset_l4/lerobot" --output-dir $NAS/init_l4 --steps 6000 > /root/l4_init.log 2>&1
grep -q INIT_OK /root/l4_init.log || { echo "!!! l4 init fail"; exit 1; }
echo "=== [l4] 加速训6k $(date) ==="
"$PY" train_cached_v5.py --resume $NAS/init_l4/checkpoints/last --repo-id local/quaddle_l4 \
    --dataset-root "$QD/outputs/dataset_l4/lerobot" --cache $NAS/cache_l4 --steps 6000 --decay 6000 \
    --batch 32 --save_freq 6000 --workers 4 --compile --output_dir $NAS/runs/l4 > /root/l4_run.log 2>&1
grep -q "training complete" /root/l4_run.log || { echo "!!! l4 train fail"; exit 1; }
echo "=== [l4] eval $(date) ==="
"$PY" eval_l4.py --ckpt $NAS/runs/l4/checkpoints/last/pretrained_model --out $NAS/runs/l4/rollout_l4.gif > /root/l4_eval.log 2>&1
grep -E 'L4_SUCCESS' /root/l4_eval.log
echo "=== [l4] 完成 $(date) ==="