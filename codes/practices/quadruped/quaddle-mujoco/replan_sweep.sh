#!/bin/bash
# 重规划间隔对比：在最佳 18k checkpoint 上 replan=5/10（20eps），对比默认25
QD=/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco
PY=/workspace/leroenv/bin/python
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
cd "$QD"
CK=outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model
for R in 5 10; do
  echo "=== replan $R ==="
  "$PY" eval_rollout_v5.py --episodes 20 --seed 7000 --replan-steps "$R" \
      --ckpt "$CK" --out /dataset/mtl_data/quaddle_vla/replan_$R.gif 2>&1 | grep -E "成功率"
done
echo "=== replan sweep done ==="