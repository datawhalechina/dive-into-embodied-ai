# LeRobot ACT：ALOHA Transfer Cube

使用 LeRobot 0.6.1 在 ALOHA Transfer Cube 人类示教数据上训练 ACT。脚本提供 CSV 指标、WebP 训练曲线、定期 checkpoint、断点续训和墙钟时限。

完整教程、实测曲线、20 回合评估和成功 GIF 见：

[`docs/practices/vla/act/index.md`](../../../../docs/practices/vla/act/index.md)

## 环境

```bash
cd codes/practices/vla/act
uv sync --frozen
```

## 冒烟测试

```bash
uv run python train_act.py \
  --episodes 0 \
  --steps 2 \
  --batch-size 2 \
  --device cuda \
  --amp \
  --log-every 1 \
  --plot-every 1 \
  --output-dir outputs/act_smoke
```

## 50k 正式训练

```bash
mkdir -p outputs/act_aloha_transfer_50k

uv run python -u train_act.py \
  --steps 50000 \
  --batch-size 8 \
  --num-workers 4 \
  --chunk-size 100 \
  --device cuda \
  --amp \
  --max-hours 2 \
  --log-every 100 \
  --plot-every 1000 \
  --save-every 10000 \
  --output-dir outputs/act_aloha_transfer_50k \
  2>&1 | tee outputs/act_aloha_transfer_50k/train.log
```

本机 RTX 4080 SUPER 实测约 39 分钟完成 50k，20 个 MuJoCo 回合成功率为 50%。

## 20 回合评估

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=outputs/act_aloha_transfer_50k/checkpoints/step_050000 \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval_act_50k_20ep
```

`--eval.batch_size=1` 可避开 LeRobot 0.6.1 异步 worker 中 `gym_aloha` 未注册的问题。
