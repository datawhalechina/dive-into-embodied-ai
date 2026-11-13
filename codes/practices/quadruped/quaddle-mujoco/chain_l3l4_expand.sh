#!/bin/bash
# L3/L4 扩充重训：各600条(后方过采样) → 缓存 → init → 训12k → eval
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"

echo "=== [L3-expand] 采集 600 eps $(date) ==="
"$PY" collect_l3.py --episodes 600 --repo-id local/quaddle_l3 \
    --root "$QD/outputs/dataset_l3" --seed 2100 --rear-oversample 0.5 > /root/l3_expand_collect.log 2>&1
grep -q "L3_DONE" /root/l3_expand_collect.log || { echo "!!! L3 collect fail"; exit 1; }

echo "=== [L4-expand] 采集 600 eps $(date) ==="
"$PY" collect_l4.py --episodes 600 --repo-id local/quaddle_l4 \
    --root "$QD/outputs/dataset_l4" --seed 3100 --rear-oversample 0.5 > /root/l4_expand_collect.log 2>&1
grep -q "L4_DONE" /root/l4_expand_collect.log || { echo "!!! L4 collect fail"; exit 1; }

echo "=== [L3-expand] 建缓存 $(date) ==="
rm -rf $NAS/cache_l3
"$PY" build_cache_v5.py --out $NAS/cache_l3 --repo-id local/quaddle_l3 \
    --dataset-root "$QD/outputs/dataset_l3/lerobot" > /root/l3_expand_cache.log 2>&1
grep -q DONE /root/l3_expand_cache.log || { echo "!!! L3 cache fail"; exit 1; }

echo "=== [L4-expand] 建缓存 $(date) ==="
rm -rf $NAS/cache_l4
"$PY" build_cache_v5.py --out $NAS/cache_l4 --repo-id local/quaddle_l4 \
    --dataset-root "$QD/outputs/dataset_l4/lerobot" > /root/l4_expand_cache.log 2>&1
grep -q DONE /root/l4_expand_cache.log || { echo "!!! L4 cache fail"; exit 1; }

echo "=== [L3-expand] init + 训 12k $(date) ==="
"$PY" init_1d_v6.py --repo-id local/quaddle_l3 \
    --dataset-root "$QD/outputs/dataset_l3/lerobot" \
    --output-dir $NAS/init_l3 --steps 12000 > /root/l3_expand_init.log 2>&1
grep -q INIT_OK /root/l3_expand_init.log || { echo "!!! L3 init fail"; exit 1; }
"$PY" train_cached_v5.py --resume $NAS/init_l3/checkpoints/last \
    --repo-id local/quaddle_l3 --dataset-root "$QD/outputs/dataset_l3/lerobot" \
    --cache $NAS/cache_l3 --steps 12000 --decay 12000 \
    --batch 32 --save_freq 6000 --workers 4 --compile \
    --output_dir $NAS/runs/l3_expand > /root/l3_expand_train.log 2>&1
grep -q "training complete" /root/l3_expand_train.log || { echo "!!! L3 train fail"; exit 1; }

echo "=== [L4-expand] init + 训 12k $(date) ==="
"$PY" init_1d_v6.py --repo-id local/quaddle_l4 \
    --dataset-root "$QD/outputs/dataset_l4/lerobot" \
    --output-dir $NAS/init_l4 --steps 12000 > /root/l4_expand_init.log 2>&1
grep -q INIT_OK /root/l4_expand_init.log || { echo "!!! L4 init fail"; exit 1; }
"$PY" train_cached_v5.py --resume $NAS/init_l4/checkpoints/last \
    --repo-id local/quaddle_l4 --dataset-root "$QD/outputs/dataset_l4/lerobot" \
    --cache $NAS/cache_l4 --steps 12000 --decay 12000 \
    --batch 32 --save_freq 6000 --workers 4 --compile \
    --output_dir $NAS/runs/l4_expand > /root/l4_expand_train.log 2>&1
grep -q "training complete" /root/l4_expand_train.log || { echo "!!! L4 train fail"; exit 1; }

echo "=== [L3-expand] eval $(date) ==="
"$PY" eval_l3.py --ckpt $NAS/runs/l3_expand/checkpoints/last/pretrained_model \
    --out $NAS/runs/l3_expand/quaddle_L3_avoid_expand.gif > /root/l3_expand_eval.log 2>&1
cat /root/l3_expand_eval.log

echo "=== [L4-expand] eval $(date) ==="
"$PY" eval_l4.py --ckpt $NAS/runs/l4_expand/checkpoints/last/pretrained_model \
    --out $NAS/runs/l4_expand/quaddle_L4_cond_expand.gif > /root/l4_expand_eval.log 2>&1
cat /root/l4_expand_eval.log

echo "=== [L3/L4-expand] 全部完成 $(date) ==="