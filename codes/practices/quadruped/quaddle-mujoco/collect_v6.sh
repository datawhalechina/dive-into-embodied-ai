#!/bin/bash
# v6 数据采集：#3 单通道档位动作（1D mode）+ #4 扩量恢复数据（warmup 15-30s + 防翻）
# 4 worker 常规（各 63）+ 4 worker 恢复（各 25）→ aggregate 成 reg(252) 与 full(352)
# 用法: nohup bash collect_v6.sh > /root/collect_v6.log 2>&1 &
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"

echo "=== [collect] 启动 8 worker（4 常规 + 4 恢复）$(date) ==="
for w in 0 1 2 3; do
  "$PY" collect_v5.py --episodes 63 --action-1d --seed 500$w \
    --repo-id local/quaddle_nav_v6_w$w --root "$QD/outputs/dataset_v6_w$w" \
    > /root/collect_v6_w$w.log 2>&1 &
done
for w in 0 1 2 3; do
  "$PY" collect_v5.py --episodes 25 --action-1d --recovery 30 --warmup-min 15 --warmup-safety \
    --seed 600$w --repo-id local/quaddle_nav_v6_r$w --root "$QD/outputs/dataset_v6_r$w" \
    > /root/collect_v6_r$w.log 2>&1 &
done
wait
echo "=== [collect] 8 worker 结束 $(date) ==="
for w in 0 1 2 3; do
  nw=$(grep -c '✓' /root/collect_v6_w$w.log); nr=$(grep -c '✓' /root/collect_v6_r$w.log)
  echo "worker $w: regular $nw / recovery $nr"
  if [ "$nw" -lt 50 ]; then echo "!!! 常规 worker $w 不足 50"; exit 1; fi
  if [ "$nr" -lt 15 ]; then echo "!!! 恢复 worker $w 不足 15"; exit 1; fi
done

echo "=== [collect] aggregate reg(252) + full(352) $(date) ==="
"$PY" - <<'EOF' > /root/merge_v6.log 2>&1
from lerobot.datasets.aggregate import aggregate_datasets
QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
w = [(f"local/quaddle_nav_v6_w{i}", f"{QD}/outputs/dataset_v6_w{i}/lerobot") for i in range(4)]
r = [(f"local/quaddle_nav_v6_r{i}", f"{QD}/outputs/dataset_v6_r{i}/lerobot") for i in range(4)]
aggregate_datasets([x[0] for x in w], "local/quaddle_nav_v6_reg",
                   roots=[x[1] for x in w], aggr_root=f"{QD}/outputs/dataset_v6_reg",
                   concatenate_videos=False)
aggregate_datasets([x[0] for x in w + r], "local/quaddle_nav_v6_full",
                   roots=[x[1] for x in w + r], aggr_root=f"{QD}/outputs/dataset_v6_full",
                   concatenate_videos=False)
print("MERGE_OK")
EOF
grep -q MERGE_OK /root/merge_v6.log || { echo "!!! 合并失败；cat /root/merge_v6.log"; exit 1; }
echo "=== [collect] 完成 reg=$(ls $QD/outputs/dataset_v6_reg/meta/info.json) full=$(ls $QD/outputs/dataset_v6_full/meta/info.json) $(date) ==="