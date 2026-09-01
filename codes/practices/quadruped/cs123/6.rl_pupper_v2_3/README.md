# Pupper RL 步态控制 v2.3

[`6.rl_pupper_v2_2`](../6.rl_pupper_v2_2/) 的增量迭代：**新增第三阶段 trot 步态条件化微调**，给涌现步态加上"节拍器"先验——这是 v2.2 之后与手写开环步态观感差距的主要来源（节律不规整）。

做法：把 v2.2 的 48 维最佳策略迁移到 53 维（追加 trot one-hot 与相位 sin/cos，新增输入权重置零），只训练 trot 一种步态，用对角接触时序奖励（权重 4.0）监督四脚按固定周期触地。环境的观测、奖励、动作滤波、扰动与 v2.2 完全一致。另有两处仅影响演示渲染：GIF 12 → 20 fps、相机跟随加一阶平滑。

版本关系：[`6.rl_pupper`](../6.rl_pupper/)（v1 基线）→ [`6.rl_pupper_v2`](../6.rl_pupper_v2/)（全面优化）→ [`6.rl_pupper_v2_1`](../6.rl_pupper_v2_1/)（钉腿惩罚）→ [`6.rl_pupper_v2_2`](../6.rl_pupper_v2_2/)（动作滤波）→ 本目录（trot 节律）。

代码复用课程公共资源 `../assets/mjcfs/pupper_v3.xml`，不在本目录重复存放 MJCF 和 mesh。

## 任务目标

训练一个接收目标速度 `(vx, vy, wz)` 的四足低层控制策略：`vx` 控制前后速度，`vy` 控制横移速度，`wz` 控制转向角速度。PPO 根据速度指令和机器人状态输出 12 个关节位置残差，由 PD 控制器执行。

任务成功不只是“机器人会走”，还要求实际速度随指令变化，并在站立、前进、后退和转向时保持稳定。默认配置不指定必须使用 walk 或 trot，腿部协调模式由奖励函数自行形成。打开 `gait.enabled` 后，任务会进一步要求策略按指定的 `walk`、`trot` 或 `pace` 接触时序运动。

训练时命令在回合内每 250 步（5 秒）重采样一次，并以 10% 概率采样零命令，使策略同时学会命令切换和安静站立。训练环境默认开启开局随机初速度和偶发 kick 扰动（评估时关闭），帮助策略跳出"站着不动"的局部最优并学会受扰恢复。

## 环境准备

在 `codes/practices/quadruped/cs123` 目录执行：

```bash
uv sync
```

## 冒烟测试

```bash
uv run pytest -q 6.rl_pupper_v2_3/tests
uv run python 6.rl_pupper_v2_3/pupper_env.py
```

## 训练策略

前两阶段直接复用 v2.2 的训练产物（本目录环境与其完全一致），只需运行第三阶段 trot 条件化微调：

```bash
uv run python 6.rl_pupper_v2_3/train.py --config 6.rl_pupper_v2_3/configs/finetune_gait.yaml
```

迁移源默认指向 `../6.rl_pupper_v2_2/reports/raw/checkpoints/pupper_ppo_8000000_steps.zip`（仓库已归档）。若要完整从零复现三阶段：

```bash
uv run python 6.rl_pupper_v2_3/train.py --config 6.rl_pupper_v2_3/configs/bootstrap_walk.yaml
uv run python 6.rl_pupper_v2_3/train.py --config 6.rl_pupper_v2_3/configs/smooth_finetune.yaml
# 然后把 finetune_gait.yaml 的 transfer_from 改为 outputs/smooth_finetune_8m/pupper_ppo.zip 再运行它
```

先用短训练确认流程正常：

```bash
uv run python 6.rl_pupper_v2_3/train.py --timesteps 100000 --n-envs 4
```

命令行参数会覆盖 YAML 中 `run` 下的对应字段。正式训练可直接修改 YAML，也可以临时覆盖：

```bash
uv run python 6.rl_pupper_v2_3/train.py \
  --timesteps 20000000 \
  --n-envs 8 \
  --tensorboard
```

加载另一份配置：

```bash
uv run python 6.rl_pupper_v2_3/train.py --config 6.rl_pupper_v2_3/configs/train.yaml
```

配置目录提供的实验：

- `bootstrap_walk.yaml` + `smooth_finetune.yaml`：推荐的两阶段课程，先学走再打磨。
- `train_no_gait.yaml`：从零单阶段训练 48 维速度条件策略。
- `train_gait.yaml`：从零单阶段训练 53 维 gait 条件策略，和上一项用于公平对照。
- `finetune_gait.yaml`：从本目录 48 维最佳策略迁移到 53 维 gait 策略并微调 5M 步。

迁移微调命令：

```bash
uv run python 6.rl_pupper_v2_3/train.py \
  --config 6.rl_pupper_v2_3/configs/finetune_gait.yaml
```

迁移时保留旧输入层已有列的权重，将新加入的观测列（机身线速度和 gait 特征）权重置零。`resume_from` 用于同结构断点续训，`transfer_from` 用于把窄观测策略迁移到宽观测策略（如旧 45 维迁移到 48 或 53 维），二者不能同时设置。

MuJoCo 仿真仍在 CPU 上运行。默认配置使用 `device: cpu`；Apple Silicon 可改成 `mps`，或临时传入 `--device mps`。SB3 的 `auto` 在没有 CUDA 时会回退 CPU，不会自动选择 MPS。训练产物写入 `6.rl_pupper_v2_3/outputs/pupper_ppo.zip`。

启用 TensorBoard 后，在另一个终端查看曲线：

```bash
uv run tensorboard --logdir 6.rl_pupper_v2_3/outputs/tb
```

断点续训：

```bash
uv run python 6.rl_pupper_v2_3/train.py \
  --timesteps 20000000 \
  --checkpoint 6.rl_pupper_v2_3/outputs/pupper_ppo.zip
```

YAML 分为五组：

- `run`：随机种子、训练步数、并行环境、设备、输出和续训路径。
- `environment`：单回合最大步数。
- `gait`：开关、可用 gait、接触奖励权重和回合内切换间隔。
- `ppo`：网络结构和 PPO 超参数。
- `checkpoint`：中间保存间隔、文件前缀和最终文件名。

## 评估策略

```bash
uv run python 6.rl_pupper_v2_3/evaluate.py
```

评估会生成：

- `6.rl_pupper_v2_3/outputs/demo.gif`
- `6.rl_pupper_v2_3/outputs/velocity_tracking.webp`

Linux 无头环境渲染失败时，可设置 `MUJOCO_GL=egl` 后重试。

## 控制设计

- 观测：基础模式为 48 维（角速度、重力方向、命令、关节角差、关节速度、上一步动作、机身线速度）；gait 模式在末尾追加三维 gait one-hot 和二维周期相位，共 53 维。机身线速度在真机上需要状态估计器提供，这里为降低仿真训练难度直接读取。
- 动作：12 维关节位置残差，由 PD 位置伺服器执行。
- 奖励：线速度与角速度跟踪、保持竖直、力矩、关节加速度、动作平滑、足端腾空时间、钉腿超时惩罚（feet_stance_time）、竖直速度与横滚俯仰角速度惩罚、足端打滑惩罚、外展角约束、零命令站立静止约束、拒绝执行命令惩罚（dont_wait）和跌倒惩罚；gait 模式额外比较目标与实际足端接触。项目集合与 `exercises/lab_6_rl_pupper` 完整实现对齐，权重按 20M 步 CPU 预算调整，并可用 `environment.reward_overrides` 按阶段覆盖。
- 命令：`vx` 范围为 `[-0.75, 0.75]`，`vy` 范围为 `[-0.5, 0.5]`，`wz` 范围为 `[-2, 2]`；回合内每 250 步重采样。
- 频率：物理仿真 250 Hz，策略控制 50 Hz。

这是便于读懂和跑通的最小版本。更完整的 540 维帧堆叠、18 项奖励、延迟和扰动随机化实现位于 `exercises/lab_6_rl_pupper`。

## 本机训练报告

M3 Ultra 上完成的两阶段课程训练（8M 步轻惩罚 bootstrap + 8M 步全额平滑惩罚微调）、checkpoint 对比、GIF、与 v1 的平稳度对比及原始数据见 [`reports/README.md`](reports/README.md)。v1 的基线报告见 [`../6.rl_pupper/reports/README.md`](../6.rl_pupper/reports/README.md)，基于 v1 45 维策略的 gait 迁移微调实验见 [`../6.rl_pupper/reports/gait_finetune/README.md`](../6.rl_pupper/reports/gait_finetune/README.md)。
