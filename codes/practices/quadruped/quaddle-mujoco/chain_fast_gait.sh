#!/bin/bash
# 快速步态重训（rank3 搜索配置）：freq2.63/ascale1.09/kp0.40/kv0.010 → 采400 → 缓存 → init → 训18k → ensemble评估 → GIF
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"
GF="--gait-freq 2.63 --gait-amp-scale 1.09 --servo-kp 0.40 --servo-kv 0.010"
CTRL="--dwell-s 0.1 --deadzone-min-deg 8 --deadzone-max-deg 20"

echo "=== [fast-gait] 采集 400 eps (rank3) $(date) ==="
"$PY" collect_v5.py --episodes 400 --repo-id local/quaddle_fast_gait \
    --root "$QD/outputs/dataset_fast_gait" --seed 8000 $GF $CTRL > /root/fast_gait_collect.log 2>&1
grep -q "完成" /root/fast_gait_collect.log || { echo "!!! collect fail"; exit 1; }

echo "=== [fast-gait] 建缓存 $(date) ==="
"$PY" build_cache_v5.py --out $NAS/cache_fast_gait --repo-id local/quaddle_fast_gait \
    --dataset-root "$QD/outputs/dataset_fast_gait/lerobot" > /root/fast_gait_cache.log 2>&1
grep -q DONE /root/fast_gait_cache.log || { echo "!!! cache fail"; exit 1; }

echo "=== [fast-gait] init $(date) ==="
"$PY" init_1d_v6.py --repo-id local/quaddle_fast_gait \
    --dataset-root "$QD/outputs/dataset_fast_gait/lerobot" \
    --output-dir $NAS/init_fast_gait --steps 18000 > /root/fast_gait_init.log 2>&1
grep -q INIT_OK /root/fast_gait_init.log || { echo "!!! init fail"; exit 1; }

echo "=== [fast-gait] 加速训 18k $(date) ==="
"$PY" train_cached_v5.py --resume $NAS/init_fast_gait/checkpoints/last \
    --repo-id local/quaddle_fast_gait \
    --dataset-root "$QD/outputs/dataset_fast_gait/lerobot" \
    --cache $NAS/cache_fast_gait --steps 18000 --decay 18000 \
    --batch 32 --save_freq 6000 --workers 4 --compile \
    --output_dir $NAS/runs/fast_gait > /root/fast_gait_train.log 2>&1
grep -q "training complete" /root/fast_gait_train.log || { echo "!!! train fail"; exit 1; }

echo "=== [fast-gait] ensemble 评估 20 eps $(date) ==="
"$PY" eval_ensemble_v5.py --ckpt $NAS/runs/fast_gait/checkpoints/last/pretrained_model \
    --episodes 20 --seed 7000 $GF --gif-fps 50 \
    --out $NAS/runs/fast_gait/quaddle_L2_nav_fastgait.gif > /root/fast_gait_eval.log 2>&1
cat /root/fast_gait_eval.log

echo "=== [fast-gait] 完成 $(date) ==="