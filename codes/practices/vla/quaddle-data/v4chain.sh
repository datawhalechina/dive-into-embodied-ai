#!/bin/bash
# Quaddle VLA v4 全链接力：等采集结束 → 训练 → seed 7000 评估 → fm-steps 20 对照
# → 20 episodes 扩展评估。工作台无人值守用。
# 用法: nohup bash v4chain.sh > /root/v4chain.log 2>&1 &

set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/vla/quaddle-data
PY=/workspace/leroenv/bin/python
TR=/workspace/leroenv/bin/lerobot-train
# v4 数据/评估全程 EGL GPU 渲染（2026-11-11 起，与采集光栅化器一致；修复见 /workspace/eglfix）
# HF_HOME 必须显式指定：ws-exec 登录 shell 会把 HOME 重置到 /root，离线缓存就找不到了
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface

echo "=== [chain] 等待采集结束 $(date) ==="
while pgrep -f '[c]ollect.py' > /dev/null; do sleep 60; done

SAVED=$(grep -c '✓' /root/collect_v4.log)
echo "=== [chain] 采集完成: ${SAVED} 条（EGL）==="
if [ "$SAVED" -lt 400 ]; then
    echo "!!! 采集不足 400 条，中止链"; exit 1
fi
# 释放 pip 缓存（checkpoint 需要磁盘）
rm -rf /workspace/.cache/pip

echo "=== [chain] 训练 v4 开始（batch32 fp32 / 6k steps，等效 192k 样本）$(date) ==="
# 注：use_amp=true 在 lerobot 0.6.1 会走 fp16 GradScaler，对 bf16 梯度直接
# NotImplementedError（smolvlm_with_expert 已知兼容问题），故 fp32 + batch32
cd "$QD" && "$TR" \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/quaddle_nav_400v4 \
    --dataset.root=outputs/dataset/lerobot \
    --batch_size=32 --steps=6000 --save_freq=6001 \
    --policy.device=cuda --policy.push_to_hub=false \
    --num_workers=8 \
    --policy.scheduler_decay_steps=6000 \
    --output_dir=outputs/smolvla_quaddle_v4 \
    > /root/v4train.log 2>&1
TR_RC=$?
echo "=== [chain] 训练退出码 ${TR_RC} $(date) ==="
if [ $TR_RC -ne 0 ]; then
    echo "!!! 训练失败，中止链（日志 /root/v4train.log）"; exit 1
fi

CKPT=outputs/smolvla_quaddle_v4/checkpoints/last/pretrained_model
echo "=== [chain] 评估 seed 7000（3 eps，对比 v1/v3 基线）$(date) ==="
cd "$QD" && "$PY" eval_rollout_gif.py --episodes 3 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v4.gif

echo "=== [chain] 评估 fm-steps 20 对照 $(date) ==="
cd "$QD" && "$PY" eval_rollout_gif.py --episodes 3 --seed 7000 --fm-steps 20 \
    --ckpt "$CKPT" --out outputs/rollout_v4_fm20.gif

echo "=== [chain] 评估 20 episodes 扩展（统计力）$(date) ==="
cd "$QD" && "$PY" eval_rollout_gif.py --episodes 20 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v4_eval20.gif

echo "=== [chain] 全部完成 $(date) ==="
