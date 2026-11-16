#!/bin/bash
# L1 指令跟随链：建缓存 → init(2D) → 加速训6k → eval_l1(>=90%)
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"
echo "=== [l1] 建缓存 $(date) ==="
if [ -f $NAS/cache_l1/info.json ]; then echo skip; else
  "$PY" build_cache_v5.py --out $NAS/cache_l1 --repo-id local/quaddle_l1 --dataset-root "$QD/outputs/dataset_l1/lerobot" > /root/l1_cache.log 2>&1
  grep -q DONE /root/l1_cache.log || { echo "!!! l1 cache fail"; exit 1; }
fi
echo "=== [l1] init $(date) ==="
"$PY" init_1d_v6.py --repo-id local/quaddle_l1 --dataset-root "$QD/outputs/dataset_l1/lerobot" --output-dir $NAS/init_l1 --steps 6000 > /root/l1_init.log 2>&1
grep -q INIT_OK /root/l1_init.log || { echo "!!! l1 init fail"; exit 1; }
echo "=== [l1] 加速训6k $(date) ==="
"$PY" train_cached_v5.py --resume $NAS/init_l1/checkpoints/last --repo-id local/quaddle_l1 \
    --dataset-root "$QD/outputs/dataset_l1/lerobot" --cache $NAS/cache_l1 --steps 6000 --decay 6000 \
    --batch 32 --save_freq 6000 --workers 4 --compile --output_dir $NAS/runs/l1 > /root/l1_run.log 2>&1
grep -q "training complete" /root/l1_run.log || { echo "!!! l1 train fail"; exit 1; }
echo "=== [l1] eval $(date) ==="
"$PY" eval_l1.py --ckpt $NAS/runs/l1/checkpoints/last/pretrained_model > /root/l1_eval.log 2>&1
grep -E 'L1_OVERALL' /root/l1_eval.log
echo "=== [l1] 完成 $(date) ==="