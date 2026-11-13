#!/bin/bash
# Quaddle VLA v5 全链接力（《Quaddle 入门》环境）：等 4 worker 采集 → 官方 aggregate
# 合并 → 训练（batch32/6k）→ seed 7000 评估 → fm-steps 20 对照 → 20 episodes 扩展。
# 用法: nohup bash chain_v5.sh > /root/v5chain.log 2>&1 &

set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
TR=/workspace/leroenv/bin/lerobot-train
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface

echo "=== [chain] 等待 4 worker 采集结束 $(date) ==="
while pgrep -f 'collect_v5.py --episodes' > /dev/null; do sleep 60; done

for w in 0 1 2 3; do
  n=$(grep -c '✓' /root/collect_w$w.log)
  echo "=== [chain] worker $w: ${n} eps ==="
  if [ "$n" -lt 60 ]; then echo "!!! worker $w 不足 60 条，中止链"; exit 1; fi
done

echo "=== [chain] 官方 aggregate 合并 $(date) ==="
# 幂等：已合并则跳过
if ! grep -q MERGE_OK /root/merge_v5.log 2>/dev/null; then
cd "$QD" && "$PY" - <<'EOF' > /root/merge_v5.log 2>&1
from lerobot.datasets.aggregate import aggregate_datasets

repo_ids = [f"local/quaddle_nav_v5_w{i}" for i in range(4)]
roots = [
    f"/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_w{i}/lerobot"
    for i in range(4)
]
aggregate_datasets(
    repo_ids,
    "local/quaddle_nav_250v5",
    roots=roots,
    aggr_root="/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_merged",
    concatenate_videos=False,
)
print("MERGE_OK")
EOF
fi
grep -q MERGE_OK /root/merge_v5.log || { echo "!!! 合并失败（日志 /root/merge_v5.log）"; exit 1; }

# 数据集根 = aggr_root 本身（info.json 在其 meta/ 子目录，不能取 dirname）
MERGED=$QD/outputs/dataset_merged
ls "$MERGED/meta/info.json" >/dev/null 2>&1 || { echo "!!! 合并数据集缺 meta/info.json"; exit 1; }
echo "=== [chain] 合并数据集位于 $MERGED ==="

echo "=== [chain] 训练 v5 开始（batch32 fp32 / 6k steps）$(date) ==="
cd "$QD" && "$TR" \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/quaddle_nav_250v5 \
    --dataset.root="$MERGED" \
    --batch_size=32 --steps=6000 --save_freq=6001 \
    --policy.device=cuda --policy.push_to_hub=false \
    --num_workers=8 \
    --policy.scheduler_decay_steps=6000 \
    --output_dir=outputs/smolvla_quaddle_v5 \
    > /root/v5train.log 2>&1
TR_RC=$?
echo "=== [chain] 训练退出码 ${TR_RC} $(date) ==="
if [ $TR_RC -ne 0 ]; then
    echo "!!! 训练失败，中止链（日志 /root/v5train.log）"; exit 1
fi

CKPT=outputs/smolvla_quaddle_v5/checkpoints/last/pretrained_model
echo "=== [chain] 评估 seed 7000（3 eps）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v5.gif

echo "=== [chain] 评估 fm-steps 20 对照 $(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 --fm-steps 20 \
    --ckpt "$CKPT" --out outputs/rollout_v5_fm20.gif

echo "=== [chain] 评估 20 episodes 扩展（统计力）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 20 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v5_eval20.gif

echo "=== [chain] 全部完成 $(date) ==="
