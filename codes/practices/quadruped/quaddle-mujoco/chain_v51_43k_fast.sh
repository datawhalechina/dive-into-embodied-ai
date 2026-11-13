#!/bin/bash
# v5.1-43k-fast 链：从 30k 续训到 43k（~10 epochs）→ seed 7000 评估 → 20 eps（oblique 斜视 GIF）
# 用法: nohup bash chain_v51_43k_fast.sh > /root/v51_43k_fast.log 2>&1 &
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface

RESUME=/dataset/mtl_data/quaddle_vla/runs/v51_30k_fast/checkpoints/last
OUT=/dataset/mtl_data/quaddle_vla/runs/v51_43k_fast
echo "=== [chain] 缓存加速续训开始（batch32 / 30k->43k）$(date) ==="
cd "$QD" && "$PY" train_cached_v5.py \
    --resume "$RESUME" \
    --steps 43000 --decay 43000 --batch 32 --save_freq 5000 --workers 4 --compile \
    --output_dir "$OUT" \
    > /root/v51_43k_fast_train.log 2>&1
TR_RC=$?
echo "=== [chain] 训练退出码 ${TR_RC} $(date) ==="
if [ $TR_RC -ne 0 ]; then echo "!!! 训练失败，中止链"; exit 1; fi

CKPT=$OUT/checkpoints/last/pretrained_model
echo "=== [chain] 评估 seed 7000（3 eps，oblique）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 \
    --ckpt "$CKPT" --out "$OUT/rollout_v51_43k_fast.gif"

echo "=== [chain] 评估 20 episodes 统计（oblique）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 20 --seed 7000 \
    --ckpt "$CKPT" --out "$OUT/rollout_v51_43k_fast_eval20.gif"

echo "=== [chain] 全部完成 $(date) ==="