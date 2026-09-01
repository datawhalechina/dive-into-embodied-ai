---
title: Pupper Locomotion｜ROCm 强化学习运动策略
sidebar_position: 2
displayed_sidebar: practicesAmdSidebar
description: "在 AMD GPU 上用 ROCm、MuJoCo、Gymnasium 与 PPO 训练 Pupper 速度跟踪策略，并复现本机训练、基准评估与调参过程。"
---

import Figure from '@site/src/components/Figure';

# Pupper Locomotion｜ROCm 强化学习运动策略

在基础教程里，我们已经用手工轨迹让 Pupper 走出了 trot；这一章改用强化学习训练一条
完整的 **locomotion policy**。策略接收期望的前进速度、横移速度和偏航角速度
`(vx, vy, wz)`，每 20 ms 输出 12 个关节位置残差，在仿真中持续跟踪命令。

本章专门讨论 AMD Linux 路线：MuJoCo 在 CPU 上推进物理，Stable-Baselines3 的 PPO
策略网络通过 PyTorch ROCm 在 AMD GPU 上训练。代码位于：

```text
codes/practices/amd/cs123/6.rl_pupper
```

如果还不熟悉 Pupper 模型、PD 控制或 Gymnasium 环境，可以先阅读
[四足机器人策略训练基础章节](/docs/practices/quadruped/cs123/rl-gait)。本章复用其任务设计，
重点补充 ROCm 环境、AMD 实现、保守调参方法，以及一组 Radeon AI PRO R9700 的真实结果。

## 本章目标

- 理解 ROCm、HIP、PyTorch 与 `torch.cuda` 设备接口之间的关系；
- 能确认 PPO 确实运行在 AMD GPU 上，而不是静默回退 CPU；
- 能解释 48 维观测、12 维 PD 残差动作和最终 12 项奖励；
- 能完成短训练、2000 万步正式训练、断点续训和 TensorBoard 监控；
- 能用固定命令、固定种子评估存活率、速度 RMSE 和姿态；
- 能从真实训练曲线判断“训练不够”和“目标设计有问题”的区别；
- 明确当前策略仍缺少 Domain Randomization，不能直接部署到真机。

## 1. 任务与计算分工

策略学习的命令范围如下：

| 指令 | 含义 | 训练范围 |
| --- | --- | --- |
| `vx` | 机身坐标系前后速度 | `[-0.75, 0.75] m/s` |
| `vy` | 机身坐标系横向速度 | `[-0.5, 0.5] m/s` |
| `wz` | 绕机身竖直轴的角速度 | `[-2.0, 2.0] rad/s` |

一次训练迭代的数据流可以概括为：

```text
8 个 MuJoCo 子进程（CPU）
        │  observation / reward
        ▼
Stable-Baselines3 rollout buffer
        │  batch
        ▼
PPO MLP 前向、反向与参数更新（ROCm GPU）
        │  12 维 action
        └──────────────────────────► MuJoCo 子进程
```

这里的“ROCm 训练”不等于“整个仿真都在 GPU”。各部分的实际设备是：

| 工作 | 设备 | 原因 |
| --- | --- | --- |
| MuJoCo 碰撞、动力学、积分 | CPU | 当前环境使用原生 MuJoCo Python API |
| 8 个环境并行采样 | CPU 多进程 | `SubprocVecEnv` 提高样本吞吐 |
| PPO 策略与价值网络 | AMD GPU | PyTorch ROCm 张量与反向传播 |
| 离屏渲染 | Radeon 图形栈 | 无头环境使用 EGL |

因此训练速度同时受 CPU 环境吞吐和 GPU 更新速度影响。当前策略只是两层 256 单元的
MLP，GPU 利用率不会像视觉模型或 GPU 物理仿真那样持续接近 100%，这不代表 ROCm
没有生效。

## 2. ROCm 是什么

[ROCm](https://rocm.docs.amd.com/) 是 AMD 面向 GPU 高性能计算和机器学习的软件栈，
包括驱动接口、HIP 运行时、编译器和数学库等组件。HIP 提供可移植的 C++ 接口，
PyTorch 的 ROCm 构建正是通过 HIP 在 AMD GPU 上执行算子。

### 2.1 为什么设备仍叫 `cuda:0`

PyTorch 为减少上层框架迁移成本，在 ROCm 构建中有意复用 `torch.cuda` API。因此：

```python
device = torch.device("cuda:0")
```

在 ROCm PyTorch 中表示第 0 张 **AMD GPU**，并不意味着机器安装了 NVIDIA CUDA。
`torch.device("rocm")` 和 `torch.device("hip")` 反而不是合法写法。判断当前后端时应同时检查：

```python
torch.cuda.is_available()  # 是否存在 PyTorch 可用的 GPU
torch.version.hip          # ROCm 构建会返回 HIP 版本
torch.version.cuda         # ROCm 构建通常为 None
```

这个行为由
[PyTorch HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html)
明确规定。

### 2.2 系统 ROCm 与 Python wheel 是两层

初次配置时最容易把下面两层混在一起：

1. **系统层**：Linux、AMDGPU 驱动、`/dev/kfd`、HIP runtime；
2. **项目层**：`.venv` 里的 ROCm 版 PyTorch、Triton 和 SB3。

`uv sync` 只负责第二层，不会替你安装或修改内核驱动。安装系统 ROCm 前，先在
[ROCm Linux 兼容性表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
核对 GPU、LLVM target、发行版和内核组合，再按
[ROCm Linux 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
操作。不能只凭“显卡来自 AMD”就假设官方 wheel 一定支持。

### 2.3 检查系统设备与权限

先确认计算设备节点存在：

```bash
ls -l /dev/kfd /dev/dri/renderD*
```

再检查 ROCm 是否正确枚举 GPU：

```bash
rocminfo | grep -E 'Name:|gfx[0-9]+'
rocm-smi --showproductname --showuse --showmemuse
```

`rocminfo` 中应出现目标卡的 `gfx` 架构，例如本次 R9700 是 `gfx1201`。如果设备节点
存在但普通用户无权访问，检查当前用户是否属于 `render` 和 `video` 组：

```bash
groups
```

需要修改组权限时，应遵循 AMD 安装文档并在重新登录后复查。不要用 `sudo` 运行训练
来掩盖权限问题，否则虚拟环境、输出文件和设备访问会变得更难维护。

### 2.4 安装项目依赖

以下命令都从 AMD 课程代码目录执行：

```bash
cd codes/practices/amd/cs123
uv sync --frozen
```

当前项目把 PyTorch 固定到官方 ROCm 7.0 wheel，并把 ROCm index 标记为显式源：

```toml
[project]
dependencies = [
    "torch==2.10.0+rocm7.0",
    "triton-rocm==3.6.0; sys_platform == 'linux'",
]

[tool.uv.sources]
torch = { index = "pytorch-rocm" }
triton-rocm = { index = "pytorch-rocm" }

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.0"
explicit = true
```

这样 `torch` 和 `triton-rocm` 只从指定的 PyTorch ROCm 仓库解析，其余依赖仍从常规
Python 包索引解析。`uv` 的索引行为可参考
[Package indexes](https://docs.astral.sh/uv/concepts/indexes/)。

:::note[版本匹配]

上面的版本是本仓库这次实验的锁定组合，不是所有机器都应永远照抄的“最新版”。更新
ROCm、驱动或 PyTorch 时，应重新检查 AMD 兼容矩阵和 PyTorch wheel 索引，并重新生成、
测试 `uv.lock`。

:::

### 2.5 做一次真正的 HIP 张量自检

不能只看 `torch.cuda.is_available()`，因为它在 CUDA 和 ROCm 构建中都可能为真。运行：

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))

assert torch.version.hip is not None
assert torch.cuda.is_available()
x = torch.tensor([1.0, 2.0, 3.0], device="cuda:0", requires_grad=True)
(x.square().sum()).backward()
torch.cuda.synchronize()
print("tensor:", x.device, "grad:", x.grad)
PY
```

本机实测的关键输出为：

```text
torch: 2.10.0+rocm7.0
HIP: 7.0.51831
CUDA: None
available: True
device: AMD Radeon AI PRO R9700
tensor: cuda:0
```

### 2.6 训练脚本拒绝 CPU 回退

`train.py` 和 `evaluate.py` 都调用 `require_rocm_device()`。它会依次检查 HIP 构建、
GPU 可用性和一个实际张量计算；任一步失败就抛出异常：

```python
def require_rocm_device() -> str:
    if torch.version.hip is None:
        raise RuntimeError("当前 PyTorch 不含 ROCm/HIP")
    if not torch.cuda.is_available():
        raise RuntimeError("未发现可用 AMD GPU")

    device = "cuda:0"
    probe = torch.ones(1, device=device)
    assert (probe * 2).item() == 2.0
    torch.cuda.synchronize(0)
    return device
```

这比 `device="auto"` 更适合验证 AMD 教程：环境配置错误时立即停止，而不是悄悄在
CPU 上跑几个小时。

训练启动后，还可以在另一个终端监控：

```bash
watch -n 1 rocm-smi --showuse --showmemuse --showpids
```

若要从 Python 侧确认已加载策略的参数设备：

```python
print(next(model.policy.parameters()).device)
```

预期输出为 `cuda:0`。

## 3. Gymnasium 环境

环境定义在 `6.rl_pupper/pupper_env.py`，模型复用
`assets/mjcfs/pupper_v3.xml`。关键频率和控制参数是：

```python
DT_PHYSICS = 0.004       # MuJoCo：250 Hz
DT_CONTROL = 0.02        # 策略：50 Hz
KP, KD = 5.0, 0.25
ACTION_SCALE = 0.5
MAX_STEPS = 1000         # 每回合最多 20 秒
```

环境遵循 Gymnasium 五元组接口：

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

- `terminated=True`：机身高度低于 `0.10 m`，或倾斜超过约 30°；
- `truncated=True`：达到 1000 个控制步；
- `info`：保存 12 个已乘权重的奖励分量，供 TensorBoard 诊断。

每次 reset 会随机化初始 yaw，在默认站姿附近加入 `±0.05 rad` 关节噪声，并采样新的
速度命令。10% 的回合使用严格零命令，以便策略专门学习站立。

:::warning[当前不是 sim2real 环境]

当前 reset 没有随机化质量、摩擦、PD 增益、传感噪声、动作延迟或外力扰动。它首先
解决平地确定性仿真中的速度跟踪，不能直接视为可部署到真机的策略。

:::

## 4. Observation 与 Action

### 4.1 48 维观测

AMD 最终版把机身线速度直接加入了观测：

| 顺序 | 内容 | 维度 |
| ---: | --- | ---: |
| 1 | 机身坐标系线速度 | 3 |
| 2 | 机身坐标系角速度 | 3 |
| 3 | 机身坐标系重力方向 | 3 |
| 4 | 目标命令 `(vx, vy, wz)` | 3 |
| 5 | 关节角相对默认站姿的偏差 | 12 |
| 6 | 关节速度 | 12 |
| 7 | 上一步动作 | 12 |
| | **合计** | **48** |

核心拼接逻辑如下：

```python
obs = np.concatenate([
    lin_vel,
    ang_vel,
    gravity,
    self.cmd,
    joint_angles - DEFAULT_POSE,
    joint_vel,
    self.last_action,
]).astype(np.float32)
```

线速度让策略直接观察跟踪误差；重力方向不依赖绝对 yaw，适合表达机身姿态；上一动作
则帮助策略约束相邻时刻的变化。当前仍是单帧观测，没有历史堆叠或循环网络。

### 4.2 12 维 PD 残差动作

动作空间为 `[-1, 1]` 内的 12 维向量。环境把它转换为默认站姿附近的目标关节角：

```python
motor_target = np.clip(
    DEFAULT_POSE + 0.5 * action,
    JOINT_LOWERS,
    JOINT_UPPERS,
)
```

MuJoCo position actuator 再按

$$
\tau = K_p(q_{\mathrm{target}}-q)-K_d\dot q
$$

产生关节力矩。残差动作让探索围绕一个可站立姿态展开，比直接输出任意关节角稳定。
每个策略动作保持 5 个物理步。

## 5. 奖励设计

最终环境包含 12 项奖励：

```python
REWARD_WEIGHTS = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 0.8,
    "lin_vel_error": -0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "torques": -2e-4,
    "action_rate": -0.01,
    "feet_air_time": 0.2,
    "stand_still": -0.5,
    "stand_still_joint_velocity": -0.05,
    "termination": -100.0,
}
```

| 奖励项 | 作用 |
| --- | --- |
| `tracking_lin_vel` | 指数奖励，跟踪 `(vx, vy)` |
| `tracking_ang_vel` | 指数奖励，跟踪 `wz` |
| `lin_vel_error` | 连续惩罚平面线速度平方误差 |
| `lin_vel_z` | 抑制机身上下弹跳 |
| `ang_vel_xy` | 抑制横滚和俯仰角速度 |
| `orientation` | 保持机身竖直 |
| `torques` | 限制关节输出的平方和 |
| `action_rate` | 抑制相邻动作高频变化 |
| `feet_air_time` | 非静止命令下奖励合理腾空时间 |
| `stand_still` | 零命令时约束关节回到默认站姿 |
| `stand_still_joint_velocity` | 零命令时抑制关节运动 |
| `termination` | 跌倒时施加终止惩罚 |

最终版的指数跟踪项为：

$$
r_{\mathrm{lin}}=
\exp\left(-\frac{\lVert v_{xy}-v^{\mathrm{cmd}}_{xy}\rVert^2}{0.10}\right),
\qquad
r_{\mathrm{yaw}}=
\exp\left(-\frac{(\omega_z-\omega_z^{\mathrm{cmd}})^2}{0.25}\right).
$$

线速度奖励核比基础版本更窄，并额外加入 `lin_vel_error`，是为了让已经能走的策略进一步
减小平移偏差。所有加权项求和后乘 `0.02`，最终奖励裁剪为非负值。

## 6. 训练 PPO

### 6.1 先做测试和短训练

```bash
cd codes/practices/amd/cs123
uv run pytest -q 6.rl_pupper/tests
uv run python 6.rl_pupper/pupper_env.py
uv run python 6.rl_pupper/train.py \
  --timesteps 100000 \
  --n-envs 4 \
  --out 6.rl_pupper/outputs/smoke
```

短训练只验证环境、多进程、ROCm、保存和加载链路，不用于评价策略好坏。四足 locomotion
通常需要数百万到数千万环境步，100k 步摔倒或不跟踪命令都不能直接证明奖励设计失败。

### 6.2 正式训练参数

本次稳定配置为：

| 参数 | 值 |
| --- | ---: |
| 并行环境 | 8 |
| 总步数 | 20,000,000 |
| `n_steps` | 2048 |
| `batch_size` | 512 |
| `n_epochs` | 4 |
| 学习率 | `1e-4` |
| `gamma` | 0.97 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| 熵系数 | `1e-3` |
| 初始 `log_std` | -0.5 |
| `target_kl` | 0.02 |
| MLP | `[256, 256]` |

运行：

```bash
uv run python 6.rl_pupper/train.py \
  --timesteps 20000000 \
  --n-envs 8 \
  --seed 42 \
  --tensorboard \
  --out 6.rl_pupper/outputs/final_train
```

最终模型写到：

```text
6.rl_pupper/outputs/final_train/pupper_ppo.zip
```

每 100 万环境步还会保存一个 checkpoint，并在同目录写入 `training_config.json`。它记录
种子、超参数、奖励权重、PyTorch/HIP 版本和 GPU 名称，不要只保留最终 zip 而丢掉配置。

### 6.3 TensorBoard 与 ROCm 同时监控

另开两个终端：

```bash
uv run tensorboard --logdir 6.rl_pupper/outputs/final_train/tb
```

```bash
watch -n 1 rocm-smi --showuse --showmemuse --showpids
```

TensorBoard 重点查看：

- `rollout/ep_rew_mean`：平均回报是否持续改善；
- `rollout/ep_len_mean`：是否逐渐稳定完成 1000 步；
- `episodes/fall_rate`：回合结束中跌倒的比例；
- `reward_components/*`：改进来自跟踪、姿态还是惩罚项；
- `train/approx_kl`：是否经常越过 `target_kl=0.02`；
- `train/std`：探索强度是否异常发散或过早坍缩。

SB3 可能提示 MLP PPO 通常更适合 CPU。这是性能建议，不是设备错误。本教程为了验证
ROCm 明确使用 GPU；判断是否生效应看 HIP 自检、策略参数设备和 `rocm-smi`，而不是
有没有这条提示。

### 6.4 断点续训

```bash
uv run python 6.rl_pupper/train.py \
  --timesteps 5000000 \
  --n-envs 8 \
  --seed 84 \
  --tensorboard \
  --checkpoint 6.rl_pupper/outputs/final_train/pupper_ppo.zip \
  --out 6.rl_pupper/outputs/final_finetune
```

当前 checkpoint 与环境都使用 48 维观测和 12 维动作，因此可以直接继续训练。若以后
改变观测维度、动作维度或网络结构，则不能把它当作普通断点续训，需要显式做权重迁移。

### 6.5 如何避免过早调参

本次实验采用了下面的顺序：

1. 先用短训练确认代码和 ROCm 链路；
2. 让稳定配置完整运行 2000 万步；
3. 用固定命令基准定位问题，而不是只看 episode return；
4. 发现中高速能走、低速与平移精度不足后，只调整相关观测与奖励；
5. 从完整模型热启动，再训练 1000 万步，并比较 500 万与 1000 万 checkpoint。

这样可以避免把“策略尚未收敛”误判成“超参数无效”。一次只改变少数有明确诊断依据的
因素，也比同时大幅修改学习率、熵、网络和奖励更容易解释结果。

## 7. 固定基准与可视化

### 7.1 为什么不能只看训练奖励

训练奖励会随奖励权重改变，两个版本的 return 不一定可直接比较。`benchmark.py` 使用
相同的外部任务指标评估 checkpoint：

| 场景 | 命令 |
| --- | --- |
| 站立 | `(0.0, 0.0, 0.0)` |
| 前进 | `(0.5, 0.0, 0.0)` |
| 后退 | `(-0.3, 0.0, 0.0)` |
| 侧移 | `(0.0, 0.3, 0.0)` |
| 偏航 | `(0.0, 0.0, 0.8)` |

每个场景使用种子 11、22、33，确定性推理 10 秒，共 15 个回合。记录存活率、完成率、
线速度 RMSE、偏航 RMSE 和平均机身倾角。

综合分数是本项目内部的 checkpoint 排序指标：

$$
100\left[
0.4S + 0.3e^{-E_{\mathrm{trans}}/0.25}
+ 0.2e^{-E_{\mathrm{yaw}}/0.5}
+ 0.1e^{-\theta/15}
\right],
$$

其中 $S$ 是存活率，$E_{\mathrm{trans}}$ 是三个平移场景的线速度 RMSE，
$E_{\mathrm{yaw}}$ 是偏航场景 RMSE，$\theta$ 是角度制平均倾角。这个分数不是 PPO 或
四足机器人领域的通用标准，应和各项原始指标一起报告。

运行最终基准：

```bash
uv run python 6.rl_pupper/benchmark.py \
  6.rl_pupper/outputs/final_train/pupper_ppo.zip \
  --output 6.rl_pupper/outputs/final_train/benchmark_final.json
```

### 7.2 生成命令演示和速度曲线

```bash
MUJOCO_GL=egl uv run python 6.rl_pupper/evaluate.py \
  --checkpoint 6.rl_pupper/outputs/final_train/pupper_ppo.zip \
  --out 6.rl_pupper/outputs/final_visual
```

会生成：

```text
demo.gif
velocity_tracking.png
```

GIF 依次执行站立、`vx=0.5` 前进、`wz=0.5` 转向、`vx=-0.3` 后退和再次站立；速度图
分别测试 `vx=0.2/0.4/0.6 m/s`。

## 8. R9700 真实训练结果

以下结果来自 2026-08-11 的一次本机训练，是特定硬件、种子和代码版本的实验记录，
不是对任意 AMD GPU 的固定性能承诺。

### 8.1 环境与训练过程

| 项目 | 实测值 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700，gfx1201，32 GiB |
| ROCm / HIP | 7.0.51831 |
| PyTorch | 2.10.0+rocm7.0 |
| PPO 参数设备 | `cuda:0` |
| MuJoCo | 3.7.0，CPU 物理仿真 |

为了不因步数不足作出错误结论，实验保留了三个阶段：

| 阶段 | 环境步 | 用途 | 固定基准 |
| --- | ---: | --- | ---: |
| v1 冒烟基线 | 300 万 | 验证链路，探索度偏高 | 58.97 分，存活率 83.4% |
| v2 stable | 2000 万，完整训练 | 稳定 PPO 配置 | 81.09 分，存活率 100% |
| v3 precision | 从 v2 追加 1000 万，完整训练 | 收紧线速度目标 | **84.48 分，存活率 100%** |

正式两阶段墙钟时间约 77 分 40 秒：v2 约 53 分 32 秒，v3 约 24 分 08 秒。v3 在
500 万追加步时为 82.28 分，到 1000 万步升至 84.48 分，因此没有因中间结果尚可而
过早停止。

<Figure
  id="fig-amd-pupper-rocm-training"
  src={require('./figs/rocm-training-curves.webp').default}
  caption="v3 precision 的 1000 万追加步训练曲线：回报继续上升，回合长度稳定在 1000 步，KL 保持在目标线以下。"
  width={1100}
/>

<Figure
  id="fig-amd-pupper-rocm-benchmark"
  src={require('./figs/rocm-benchmark-progress.webp').default}
  caption="v3 在 500 万与 1000 万追加步的固定基准：存活率保持 100%，综合分数提高，平移和偏航 RMSE 均下降。"
  width={1100}
/>

### 8.2 最终固定基准

| 指标 | v2 stable | v3 precision | 变化 |
| --- | ---: | ---: | ---: |
| 综合分数 | 81.09 | **84.48** | +3.39 |
| 存活 / 完成率 | 100% / 100% | **100% / 100%** | 持平 |
| 全场景线速度 RMSE | 0.101 m/s | **0.062 m/s** | -38.1% |
| 平移命令 RMSE | 0.139 m/s | **0.080 m/s** | -42.3% |
| 偏航命令 RMSE | **0.130 rad/s** | 0.151 rad/s | +16.4% |
| 平均机身倾角 | **2.47°** | 3.45° | +0.98° |

v3 用少量偏航精度和姿态平稳度换来了明显更好的平移跟踪，因此综合分数更高。三个种子
的速度均值如下：

| 场景 | 目标命令 | 实际均值 |
| --- | --- | --- |
| 站立 | `(0, 0, 0)` | `vx=-0.001, vy=0.005, wz=-0.023` |
| 前进 | `vx=0.5` | `vx=0.507 m/s` |
| 后退 | `vx=-0.3` | `vx=-0.336 m/s` |
| 侧移 | `vy=0.3` | `vy=0.350 m/s` |
| 偏航 | `wz=0.8` | `wz=0.736 rad/s` |

### 8.3 成功与失败都要保留

<Figure
  id="fig-amd-pupper-rocm-velocity"
  src={require('./figs/rocm-velocity-tracking.webp').default}
  caption="最终策略的三档前向速度跟踪：0.4 和 0.6 m/s 能稳定跟踪，0.2 m/s 仍倾向原地站立。"
  width={1100}
/>

`vx=0.2 m/s` 的失败很有信息量：继续无差别增加训练步数未必能解决离散的“站立或行走”
行为。下一轮更合理的实验是增加低速命令采样密度，或单独设计低速启动与维持奖励，并
保留相同基准做对照。

命令切换演示如下。它可以连续完成站立、前进、转向和后退；固定基准的 15 个回合均
运行满 10 秒。

<Figure
  id="fig-amd-pupper-rocm-demo"
  src={require('./figs/rocm-command-demo.gif').default}
  caption="最终 v3 precision checkpoint 的命令切换演示。"
  width={640}
/>

完整参数、逐场景数据、checkpoint 哈希和复现记录见
[`TRAINING_REPORT.md`](https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/codes/practices/amd/cs123/6.rl_pupper/TRAINING_REPORT.md)。
训练产物位于本机 `6.rl_pupper/outputs/`；该目录被 `.gitignore` 排除，克隆仓库后需要
自行训练或另外获取 checkpoint。

## 9. 常见问题

### 9.1 `torch.version.hip` 是 `None`

这表示当前环境装的不是 ROCm 版 PyTorch。确认工作目录正确，并检查：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.hip)"
uv tree | grep -E 'torch|triton'
```

再核对 `pyproject.toml`、`uv.lock` 和官方 ROCm index。不要通过把脚本改成 `device=cpu`
来绕过教程验收。

### 9.2 `torch.cuda.is_available()` 为 `False`

依次检查：

1. GPU、系统和 ROCm 版本是否在官方兼容矩阵中；
2. `/dev/kfd` 与 `/dev/dri/renderD*` 是否存在；
3. `rocminfo` 能否枚举目标 GPU；
4. 当前用户是否有 `render`、`video` 组权限；
5. 系统 ROCm 与 PyTorch wheel 是否匹配。

### 9.3 多张 AMD GPU 时选错设备

先用 `rocminfo` 和 `rocm-smi` 确认索引。必要时在启动进程前限制可见设备：

```bash
HIP_VISIBLE_DEVICES=0 uv run python 6.rl_pupper/train.py --timesteps 100000
```

屏蔽后，进程内第一张可见 GPU 仍显示为 `cuda:0`。每次都用
`torch.cuda.get_device_name(0)` 复查，不要只相信物理卡号。

### 9.4 ROCm 正常但 GPU 利用率不高

这是当前架构的正常现象：MuJoCo 采样在 CPU，多进程需要把小批量观测送到一个很小的
MLP。先看总体环境步吞吐；如果增加 `n_envs` 后 FPS 提高，瓶颈仍在 CPU 采样。不要为了
让 GPU 百分比更好看而盲目扩大网络或 batch，这会改变实验本身。

### 9.5 无头服务器渲染失败

训练本身不需要渲染。只有生成 GIF 时设置：

```bash
MUJOCO_GL=egl uv run python 6.rl_pupper/evaluate.py
```

如果 EGL 仍失败，再检查系统 Mesa/AMDGPU 图形栈；这与 PyTorch HIP 张量能否计算是两个
独立问题。

## 10. 局限与下一步

当前版本已经证明 AMD ROCm 训练链路和全方向速度命令基准可用，但仍有清晰边界：

1. `vx=0.2 m/s` 低速命令仍可能选择站立；
2. 后退和侧移略有超调，偏航略有欠跟踪；
3. 观测没有历史帧、足端接触力和地形信息；
4. 足端接触由 site 高度近似，不是完整 contact pair；
5. 没有质量、摩擦、执行器、噪声、延迟、地形和外力 Domain Randomization；
6. 只完成确定性平地仿真，没有真机验证。

下一阶段建议按以下顺序推进：

1. 增加低速命令分层采样，复测 `0.1–0.4 m/s`；
2. 在不改其他条件的情况下分别处理后退、侧移和偏航偏差；
3. 一次加入一种 Domain Randomization，并保留无随机化对照；
4. 增加扰动恢复、组合命令和更长回合测试；
5. 最后再讨论 sim2real、真机限位和安全停止。

完成稳定的 locomotion policy 后，可以把它封装成 `walk`、`turn`、`stop` 等技能，
继续进入 [Pupper VLA](./vla)。

## 参考资料

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm Linux system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [PyTorch HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [uv package indexes](https://docs.astral.sh/uv/concepts/indexes/)
