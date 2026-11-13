#!/bin/bash
# v6 训练链：#3 单通道档位（Exp A=reg252）+ #3+#4 扩量恢复（Exp B=full352）
# 缓存构建 → lerobot-train 20 步生成 1D 适配 init checkpoint → 加速续训 18k → 斜视评估
# 用法: nohup bash chain_v6_train.sh > /root/v6_train.log 2>&1 &
set -x
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
TR=/workspace/leroenv/bin/lerobot-train
NAS=/dataset/mtl_data/quaddle_vla
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"

for TAG in reg full; do
  DSROOT=$QD/outputs/dataset_v6_$TAG
  REPO=local/quaddle_nav_v6_$TAG
  ls "$DSROOT/meta/info.json" >/dev/null 2>&1 || { echo "!!! 缺数据集 $DSROOT"; exit 1; }

  echo "=== [v6/$TAG] 构建图像嵌入缓存 $(date) ==="
  if [ -f $NAS/cache_v6_$TAG/info.json ]; then
    echo "=== [v6/$TAG] 缓存已存在，跳过构建 ==="
  else
    "$PY" build_cache_v5.py --out $NAS/cache_v6_$TAG --repo-id $REPO --dataset-root "$DSROOT" \
        > /root/v6_cache_$TAG.log 2>&1
    grep -q DONE /root/v6_cache_$TAG.log || { echo "!!! 缓存失败 $TAG"; exit 1; }
  fi

  echo "=== [v6/$TAG] make_policy 生成 1D 适配 init checkpoint（step0，不前向）$(date) ==="
  "$PY" init_1d_v6.py --repo-id $REPO --dataset-root "$DSROOT" --output-dir $NAS/init_v6_$TAG --steps 18000 \
      > /root/v6_init_$TAG.log 2>&1
  grep -q INIT_OK /root/v6_init_$TAG.log || { echo "!!! init 失败 $TAG"; exit 1; }

  echo "=== [v6/$TAG] 加速续训 18k $(date) ==="
  "$PY" train_cached_v5.py --resume $NAS/init_v6_$TAG/checkpoints/last \
      --repo-id $REPO --dataset-root "$DSROOT" --cache $NAS/cache_v6_$TAG \
      --steps 18000 --decay 18000 --batch 32 --save_freq 6000 --workers 4 --compile \
      --output_dir $NAS/runs/v6_${TAG}_18k > /root/v6_run_$TAG.log 2>&1
  grep -q "training complete" /root/v6_run_$TAG.log || { echo "!!! 训练失败 $TAG"; exit 1; }

  CKPT=$NAS/runs/v6_${TAG}_18k/checkpoints/last/pretrained_model
  echo "=== [v6/$TAG] 评估 seed 7000（3 eps，oblique）$(date) ==="
  "$PY" eval_rollout_v5.py --episodes 3 --seed 7000 --ckpt "$CKPT" --out $NAS/runs/v6_${TAG}_18k/rollout_seed7000.gif
  echo "=== [v6/$TAG] 评估 20 episodes（oblique）$(date) ==="
  "$PY" eval_rollout_v5.py --episodes 20 --seed 7000 --ckpt "$CKPT" --out $NAS/runs/v6_${TAG}_18k/rollout_eval20.gif
done
echo "=== [v6] 全部完成 $(date) ==="