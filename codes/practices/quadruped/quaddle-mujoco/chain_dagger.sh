#!/bin/bash
# DAgger 闭环恢复重训链：合并 reg252(2D)+dagger → 建缓存 → 从18k加速续训 → replan5评估
# 用法: nohup bash chain_dagger.sh > /root/dagger_train.log 2>&1 &
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"

echo "=== [dagger] 合并 reg252 + dagger $(date) ==="
"$PY" - <<'EOF' > /root/merge_dagger.log 2>&1
from lerobot.datasets.aggregate import aggregate_datasets
QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
repos = [f"local/quaddle_nav_v5_w{i}" for i in range(4)] + ["local/quaddle_nav_dagger"]
roots = [f"{QD}/outputs/dataset_w{i}/lerobot" for i in range(4)] + [f"{QD}/outputs/dataset_dagger/lerobot"]
aggregate_datasets(repos, "local/quaddle_nav_dagger_merged", roots=roots,
                   aggr_root=f"{QD}/outputs/dataset_dagger_merged", concatenate_videos=False)
print("MERGE_OK")
EOF
grep -q MERGE_OK /root/merge_dagger.log || { echo "!!! merge 失败"; exit 1; }

echo "=== [dagger] 建缓存 $(date) ==="
if [ -f $NAS/cache_dagger/info.json ]; then echo "cache exists, skip"; else
  "$PY" build_cache_v5.py --out $NAS/cache_dagger --repo-id local/quaddle_nav_dagger_merged \
      --dataset-root "$QD/outputs/dataset_dagger_merged" > /root/dagger_cache.log 2>&1
  grep -q DONE /root/dagger_cache.log || { echo "!!! cache 失败"; exit 1; }
fi

echo "=== [dagger] 从18k加速续训到24k $(date) ==="
"$PY" train_cached_v5.py --resume "$QD/outputs/smolvla_quaddle_v51_18k/checkpoints/last" \
    --repo-id local/quaddle_nav_dagger_merged --dataset-root "$QD/outputs/dataset_dagger_merged" \
    --cache $NAS/cache_dagger --steps 24000 --decay 24000 --batch 32 --save_freq 6000 \
    --workers 4 --compile --output_dir $NAS/runs/dagger_24k > /root/dagger_run.log 2>&1
grep -q "training complete" /root/dagger_run.log || { echo "!!! train 失败"; exit 1; }

CKPT=$NAS/runs/dagger_24k/checkpoints/last/pretrained_model
echo "=== [dagger] 评估 seed7000 + 20eps (replan5) $(date) ==="
"$PY" eval_rollout_v5.py --episodes 3 --seed 7000 --ckpt "$CKPT" --out $NAS/runs/dagger_24k/rollout_seed7000.gif
"$PY" eval_rollout_v5.py --episodes 20 --seed 7000 --ckpt "$CKPT" --out $NAS/runs/dagger_24k/rollout_eval20.gif
echo "=== [dagger] 全部完成 $(date) ==="