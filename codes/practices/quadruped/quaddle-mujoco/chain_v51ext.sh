#!/bin/bash
# v5.1-ext 延长训练链：18k 步（4.2 epochs，检验欠拟合假设）→ seed 7000 评估 → 20 eps 统计
# 用法: nohup bash chain_v51ext.sh > /root/v51ext.log 2>&1 &

set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
TR=/workspace/leroenv/bin/lerobot-train
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface

MERGED=$QD/outputs/dataset_merged51
ls "$MERGED/meta/info.json" >/dev/null 2>&1 || { echo "!!! 合并数据集不存在"; exit 1; }

echo "=== [chain] 延长训练开始（batch32 fp32 / 18k steps）$(date) ==="
cd "$QD" && "$TR" \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/quaddle_nav_352v51 \
    --dataset.root="$MERGED" \
    --batch_size=32 --steps=18000 --save_freq=18001 \
    --policy.device=cuda --policy.push_to_hub=false \
    --num_workers=8 \
    --policy.scheduler_decay_steps=18000 \
    --output_dir=outputs/smolvla_quaddle_v51_18k \
    > /root/v51ext_train.log 2>&1
TR_RC=$?
echo "=== [chain] 训练退出码 ${TR_RC} $(date) ==="
if [ $TR_RC -ne 0 ]; then
    echo "!!! 训练失败，中止链（日志 /root/v51ext_train.log）"; exit 1
fi

CKPT=outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model
echo "=== [chain] 评估 seed 7000（3 eps）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v51_18k.gif

echo "=== [chain] 评估 20 episodes 统计 $(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 20 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v51_18k_eval20.gif

echo "=== [chain] 全部完成 $(date) ==="
