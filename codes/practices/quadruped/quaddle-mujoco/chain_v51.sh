#!/bin/bash
# v5.1 过夜链：等 4 个恢复数据 worker → 8 数据集合并（252 常规 + ~100 恢复）
# → 训练 batch32/6k → seed 7000 评估 → fm20 → 20 eps 统计
# 用法: nohup bash chain_v51.sh > /root/v51chain.log 2>&1 &

set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
TR=/workspace/leroenv/bin/lerobot-train
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface

echo "=== [chain] 等待恢复数据 worker 结束 $(date) ==="
while pgrep -f 'collect_v5.py --episodes' > /dev/null; do sleep 60; done

for w in 0 1 2 3; do
  n=$(grep -c '✓' /root/collect_r$w.log)
  echo "=== [chain] recovery worker $w: ${n} eps ==="
  if [ "$n" -lt 20 ]; then echo "!!! worker $w 不足 20 条，中止链"; exit 1; fi
done

echo "=== [chain] 8 数据集官方 aggregate 合并 $(date) ==="
cd "$QD" && "$PY" - <<'EOF' > /root/merge_v51.log 2>&1
from lerobot.datasets.aggregate import aggregate_datasets

repo_ids, roots = [], []
for i in range(4):
    repo_ids.append(f"local/quaddle_nav_v5_w{i}")
    roots.append(f"/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_w{i}/lerobot")
for i in range(4):
    repo_ids.append(f"local/quaddle_nav_v5_r{i}")
    roots.append(f"/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_r{i}/lerobot")
aggregate_datasets(
    repo_ids,
    "local/quaddle_nav_352v51",
    roots=roots,
    aggr_root="/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_merged51",
    concatenate_videos=False,
)
print("MERGE_OK")
EOF
grep -q MERGE_OK /root/merge_v51.log || { echo "!!! 合并失败（日志 /root/merge_v51.log）"; exit 1; }

MERGED=$QD/outputs/dataset_merged51
ls "$MERGED/meta/info.json" >/dev/null 2>&1 || { echo "!!! 合并数据集缺 meta/info.json"; exit 1; }
echo "=== [chain] 合并数据集位于 $MERGED ==="

echo "=== [chain] 训练 v5.1 开始（batch32 fp32 / 6k steps）$(date) ==="
cd "$QD" && "$TR" \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/quaddle_nav_352v51 \
    --dataset.root="$MERGED" \
    --batch_size=32 --steps=6000 --save_freq=6001 \
    --policy.device=cuda --policy.push_to_hub=false \
    --num_workers=8 \
    --policy.scheduler_decay_steps=6000 \
    --output_dir=outputs/smolvla_quaddle_v51 \
    > /root/v51train.log 2>&1
TR_RC=$?
echo "=== [chain] 训练退出码 ${TR_RC} $(date) ==="
if [ $TR_RC -ne 0 ]; then
    echo "!!! 训练失败，中止链（日志 /root/v51train.log）"; exit 1
fi

CKPT=outputs/smolvla_quaddle_v51/checkpoints/last/pretrained_model
echo "=== [chain] 评估 seed 7000（3 eps）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v51.gif

echo "=== [chain] 评估 fm-steps 20 对照 $(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 --fm-steps 20 \
    --ckpt "$CKPT" --out outputs/rollout_v51_fm20.gif

echo "=== [chain] 评估 20 episodes 扩展（统计力）$(date) ==="
cd "$QD" && "$PY" eval_rollout_v5.py --episodes 20 --seed 7000 \
    --ckpt "$CKPT" --out outputs/rollout_v51_eval20.gif

echo "=== [chain] 全部完成 $(date) ==="
