# Pupper RL 步态控制

自包含的四足 Pupper 强化学习步态控制示例：Gymnasium 环境、SB3 PPO 训练与评估渲染。

代码复用课程公共资源 `../assets/mjcfs/pupper_v3.xml`，不在本目录重复存放 MJCF 和 mesh。

## 任务目标

训练一个接收目标速度 `(vx, vy, wz)` 的四足低层控制策略：`vx` 控制前后速度，`vy` 控制横移速度，`wz` 控制转向角速度。PPO 根据速度指令和机器人状态输出 12 个关节位置残差，由 PD 控制器执行。

任务成功不只是“机器人会走”，还要求实际速度随指令变化，并在站立、前进、后退和转向时保持稳定。默认配置不指定必须使用 walk 或 trot，腿部协调模式由奖励函数自行形成。打开 `gait.enabled` 后，任务会进一步要求策略按指定的 `walk`、`trot` 或 `pace` 接触时序运动。

## 环境准备

在 `codes/practices/quadruped/cs123` 目录执行：

```bash
uv sync
```

## 冒烟测试

```bash
uv run pytest -q 6.rl_pupper/tests
uv run python 6.rl_pupper/pupper_env.py
```

## 训练策略

默认从 `configs/train.yaml` 加载运行参数、PPO 超参数和 checkpoint 配置：

```bash
uv run python 6.rl_pupper/train.py
```

先用短训练确认流程正常：

```bash
uv run python 6.rl_pupper/train.py --timesteps 100000 --n-envs 4
```

命令行参数会覆盖 YAML 中 `run` 下的对应字段。正式训练可直接修改 YAML，也可以临时覆盖：

```bash
uv run python 6.rl_pupper/train.py \
  --timesteps 20000000 \
  --n-envs 8 \
  --tensorboard
```

加载另一份配置：

```bash
uv run python 6.rl_pupper/train.py --config 6.rl_pupper/configs/train.yaml
```

配置目录提供三种互不混淆的实验：

- `train_no_gait.yaml`：从零训练 45 维速度条件策略。
- `train_gait.yaml`：从零训练 50 维 gait 条件策略，和上一项用于公平对照。
- `finetune_gait.yaml`：从已有 19M 步、45 维策略迁移并微调 5M 步，用于快速验证。

迁移微调命令：

```bash
uv run python 6.rl_pupper/train.py \
  --config 6.rl_pupper/configs/finetune_gait.yaml
```

迁移时保留旧输入层前 45 列权重，将新加入的 5 列 gait 输入权重置零。`resume_from` 用于同结构断点续训，`transfer_from` 用于 45 维到 50 维迁移，二者不能同时设置。

MuJoCo 仿真仍在 CPU 上运行。默认配置使用 `device: cpu`；Apple Silicon 可改成 `mps`，或临时传入 `--device mps`。SB3 的 `auto` 在没有 CUDA 时会回退 CPU，不会自动选择 MPS。训练产物写入 `6.rl_pupper/outputs/pupper_ppo.zip`。

启用 TensorBoard 后，在另一个终端查看曲线：

```bash
uv run tensorboard --logdir 6.rl_pupper/outputs/tb
```

断点续训：

```bash
uv run python 6.rl_pupper/train.py \
  --timesteps 20000000 \
  --checkpoint 6.rl_pupper/outputs/pupper_ppo.zip
```

YAML 分为五组：

- `run`：随机种子、训练步数、并行环境、设备、输出和续训路径。
- `environment`：单回合最大步数。
- `gait`：开关、可用 gait、接触奖励权重和回合内切换间隔。
- `ppo`：网络结构和 PPO 超参数。
- `checkpoint`：中间保存间隔、文件前缀和最终文件名。

## 评估策略

```bash
uv run python 6.rl_pupper/evaluate.py
```

评估会生成：

- `6.rl_pupper/outputs/demo.gif`
- `6.rl_pupper/outputs/velocity_tracking.webp`

Linux 无头环境渲染失败时，可设置 `MUJOCO_GL=egl` 后重试。

## 控制设计

- 观测：基础模式为 45 维；gait 模式在末尾追加三维 gait one-hot 和二维周期相位，共 50 维。
- 动作：12 维关节位置残差，由 PD 位置伺服器执行。
- 奖励：线速度与角速度跟踪、保持竖直、力矩、动作平滑、足端腾空时间和跌倒惩罚；gait 模式额外比较目标与实际足端接触。
- 命令：`vx` 范围为 `[-0.75, 0.75]`，`vy` 范围为 `[-0.5, 0.5]`，`wz` 范围为 `[-2, 2]`。
- 频率：物理仿真 250 Hz，策略控制 50 Hz。

这是便于读懂和跑通的最小版本。更完整的 540 维帧堆叠、18 项奖励、延迟和扰动随机化实现位于 `exercises/lab_6_rl_pupper`。

## 本机训练报告

M3 Ultra 上完成的 20M 步基础训练、checkpoint 对比、GIF 和原始数据见 [`reports/README.md`](reports/README.md)。带 gait 的 5M 迁移微调及其接触时序评估见 [`reports/gait_finetune/README.md`](reports/gait_finetune/README.md)。
