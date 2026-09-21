# π₀.₅ + RECAP：LIBERO-10 Task 0

这组脚本为 OpenPI 增加 RECAP 复现流程：准备 RLinf 的 LeRobot 数据、训练价值模型、标注优势、微调 ACP 策略，并在 LIBERO 中评估成功率与可视化 value。10k/20k 价值模型的验证误差、数据说明和成功回合动图见[完整教程](../../../../docs/practices/vla/recap/index.md)。

## 环境

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 将目录内容复制进去；重复执行也不会生成 examples/recap/recap/
mkdir -p examples/recap
cp -a ../dive-into-embodied-ai/codes/practices/vla/recap/. examples/recap/
```

## 数据准备

```bash
export HF_LEROBOT_HOME="$PWD/data/lerobot"
export RECAP_REPO_ID="local/libero10_task0_train"
export RECAP_EVAL_REPO_ID="local/libero10_task0_eval"

HF_ENDPOINT=https://hf-mirror.com uv run python examples/recap/run.py data download
for split in sft train eval; do
  uv run python examples/recap/run.py data prepare --split "$split"
done
uv run python examples/recap/run.py stats --config-name pi05_recap_sft

for repo_id in "$RECAP_REPO_ID" "$RECAP_EVAL_REPO_ID"; do
  uv run python examples/recap/run.py annotate targets \
    --dataset-root "$HF_LEROBOT_HOME/$repo_id"
done
```

## 价值模型与优势标注

```bash
export CUDA_VISIBLE_DEVICES=6,7
uv run python examples/recap/run.py train pi05_recap_value \
  --exp-name value --num-train-steps 30000 \
  --batch-size 18 --fsdp-devices 2 \
  --save-interval 10000 --keep-period 10000 --no-wandb-enabled

VALUE_CKPT="$PWD/checkpoints/pi05_recap_value/value/10000"
uv run python examples/recap/run.py annotate evaluate \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_EVAL_REPO_ID" \
  --checkpoint "$VALUE_CKPT" --batch-size 8 --max-frames 10000 \
  --output data/recap/value_eval/10000.json

export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run python examples/recap/run.py annotate advantages \
  --dataset-root "$HF_LEROBOT_HOME/$RECAP_REPO_ID" \
  --checkpoint "$VALUE_CKPT" --batch-size 72 --num-workers 8 \
  --n-step 10 --positive-ratio 0.3
```

## ACP 微调

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset RECAP_INIT_PARAMS
uv run python examples/recap/run.py train pi05_recap_acp \
  --exp-name acp-b36 --num-train-steps 90000 \
  --batch-size 36 --fsdp-devices 4 \
  --lr-schedule.warmup-steps 3000 --lr-schedule.decay-steps 90000 \
  --save-interval 10000 --keep-period 10000 --no-wandb-enabled
```

## LIBERO 评估与 value 曲线

先按完整教程安装 `third_party/libero` 并创建 `examples/libero/.venv`。随后在 OpenPI 根目录分别打开三个终端：

```bash
# 终端 1：ACP 策略服务
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/recap/run.py serve --port 8000 \
  policy:checkpoint --policy.config pi05_recap_acp \
  --policy.dir "$PWD/checkpoints/pi05_recap_acp/acp-b36/60000"
```

```bash
# 终端 2：10k value 服务
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python examples/recap/run.py serve-value \
  --checkpoint "$PWD/checkpoints/pi05_recap_value/value/10000" --port 8001
```

```bash
# 终端 3：20 回合仿真；输出目录须是新的
PYTHONPATH="$PWD/third_party/libero" MUJOCO_GL=egl \
examples/libero/.venv/bin/python examples/recap/run.py rollout eval \
  --output data/recap/eval_acp_b36_60k_value \
  --policy-label "$PWD/checkpoints/pi05_recap_acp/acp-b36/60000" \
  --value-port 8001 --num-episodes 20
```

每次重规划都会把 value 写入 `state_*_values.jsonl`，每回合结束生成带曲线的 `*_value.mp4`；成功率记录在 `results.json`。两个成功回合视频只是示例，不能代替 20 回合成功率或与起始策略的同条件对照。
