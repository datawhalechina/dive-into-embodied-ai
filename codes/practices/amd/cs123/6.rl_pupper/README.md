# Pupper RL 步态控制

自包含的四足 Pupper 强化学习步态控制示例：Gymnasium 环境、SB3 PPO 训练与评估渲染。

代码复用课程公共资源 `../assets/mjcfs/pupper_v3.xml`，不在本目录重复存放 MJCF 和 mesh。

## 环境准备

在 `codes/practices/amd/cs123` 目录执行：

```bash
uv sync
```

## 冒烟测试

```bash
uv run pytest -q 6.rl_pupper/tests
uv run python 6.rl_pupper/pupper_env.py
```

## 训练策略

先用短训练确认流程正常：

```bash
uv run python 6.rl_pupper/train.py --timesteps 100000 --n-envs 4
```

正式训练可从 2000 万环境步开始：

```bash
uv run python 6.rl_pupper/train.py --timesteps 20000000 --n-envs 8 --tensorboard
```

MuJoCo 仿真在 CPU 上运行，PPO 网络和张量计算强制使用 ROCm 7.0。训练启动时会
执行 HIP 张量自检；ROCm 或 AMD GPU 不可用时直接报错，不会静默回退到 CPU。
训练产物写入 `6.rl_pupper/outputs/pupper_ppo.zip`。

断点续训：

```bash
uv run python 6.rl_pupper/train.py \
  --timesteps 20000000 \
  --checkpoint 6.rl_pupper/outputs/pupper_ppo.zip
```

## 评估策略

```bash
uv run python 6.rl_pupper/evaluate.py
```

评估会生成：

- `6.rl_pupper/outputs/demo.gif`
- `6.rl_pupper/outputs/velocity_tracking.png`

Linux 无头环境渲染失败时，可设置 `MUJOCO_GL=egl` 后重试。

固定命令基准：

```bash
uv run python 6.rl_pupper/benchmark.py 6.rl_pupper/outputs/pupper_ppo.zip
```

本机 R9700 的完整 ROCm 训练过程、量化结果和产物索引见
[`TRAINING_REPORT.md`](TRAINING_REPORT.md)。

## 控制设计

- 观测：48 维，包括机身线/角速度、重力方向、速度命令、关节状态和上一步动作。
- 动作：12 维关节位置残差，由 PD 位置伺服器执行。
- 奖励：线/角速度跟踪、保持竖直、垂直速度、横滚/俯仰角速度、力矩、动作平滑、
  足端腾空时间、零命令静止和跌倒惩罚。
- 命令：`vx` 范围为 `[-0.75, 0.75]`，`vy` 范围为 `[-0.5, 0.5]`，`wz` 范围为 `[-2, 2]`。
- 频率：物理仿真 250 Hz，策略控制 50 Hz。

这是便于读懂和跑通的最小版本。更完整的 540 维帧堆叠、18 项奖励、延迟和扰动随机化实现位于 `exercises/lab_6_rl_pupper`。
