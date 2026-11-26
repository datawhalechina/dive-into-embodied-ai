---
title: 2. 训练 Go2 速度策略
sidebar_position: 3
displayed_sidebar: practicesAmdSidebar
description: "使用 MuJoCo MJX 与 JAX 在单张 AMD 或 NVIDIA GPU 上完成 Unitree Go2 平地速度跟踪策略的冒烟训练、完整训练与离线渲染。"
---

# 2. 训练 Go2 速度策略

本章训练 Go2 平地速度跟踪任务：策略接收本体感知观测与三维速度指令（前向 `vx`、侧向 `vy`、偏航角速度 `wz`），输出 12 个关节的位置目标。奖励与课程设计移植自 Unitree 官方的 `unitree_rl_mjlab`（Isaac Lab 生态），物理后端换成 MuJoCo MJX，因此整个训练循环可在单张 GPU 上以 JAX 编译执行。

所有命令默认在 `codes/practices/amd/unitree-rl-mjx/` 目录下运行。

## 冒烟训练（约 20 分钟） \{#smoke-training}

先用 4000 万步确认整条链路（ROCm 机器记得带上 [上一节](./environment-setup.md#rdna4-constraints) 的 `XLA_FLAGS`）：

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  uv run python -m unitree_rl_mjx.train.go2_velocity \
  --seed 0 --num-timesteps 41943040 --out-dir runs/go2-smoke
```

训练过程中每个评估点打印一行 `steps reward`；奖励应从负值起步、数百万步内转正并持续上升（4000 万步时约 `40+`）。

## 完整训练 \{#full-training}

不带 `--num-timesteps` 即训练完整的 **9.8 亿步**公开预算：

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  uv run python -m unitree_rl_mjx.train.go2_velocity \
  --seed 0 --out-dir runs/go2-full
```

- **实测墙钟**：Radeon AI PRO R9700 在 ROCm 7.2.4 下约 **7.9 小时**，升级至 **ROCm 10.2.0 + `jax==0.11.2`** 后仅需 **4.48 小时（16,136 s，吞吐 62.4k steps/s，最终回报 `51.20`）**；NVIDIA A40 约 **5.5 小时**；不同后端训练出的策略收敛质量一致。
- **预期波动**：训练在约 **4.9 亿步**处会经历一次指令课程加宽（回报短暂回落后恢复），属预期现象。

## 输出文件 \{#outputs}

`--out-dir` 下会生成 4 个核心文件：

| 文件 | 内容 |
| :--- | :--- |
| `metrics.jsonl` | 逐评估点的回报与训练吞吐（JSON Lines） |
| `params.bin` | 训练完成的策略参数（Brax 格式，下一节用于导出 ONNX） |
| `trajectory.npz` | 训练结束后一条评估轨迹的 `qpos` 序列 |
| `run.json` | 运行清单：后端、设备、版本、墙钟、随机种子 |

## 绘制训练曲线 \{#plot-curve}

```python
import json
import matplotlib.pyplot as plt

rows = [json.loads(line) for line in open("runs/go2-smoke/metrics.jsonl")]
steps = [r["step"] for r in rows if "eval/episode_reward" in r]
reward = [r["eval/episode_reward"] for r in rows if "eval/episode_reward" in r]
plt.plot(steps, reward)
plt.xlabel("environment steps")
plt.ylabel("episode reward")
plt.show()
```

## 离线渲染训练结果 \{#render-rollout}

`trajectory.npz` 配合任务自带的场景模型即可离线渲染步态 GIF：

```python
import imageio
import mujoco
import numpy as np
from unitree_rl_mjx.envs import Go2VelocityFlat

env = Go2VelocityFlat()
qpos = np.load("runs/go2-smoke/trajectory.npz")["qpos"]

model = env.mj_model
model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
data = mujoco.MjData(model)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
camera.trackbodyid = model.body("base_link").id
camera.distance, camera.elevation, camera.azimuth = 1.6, -18, 125

renderer = mujoco.Renderer(model, height=540, width=960)
frames = []
for q in qpos[::4]:  # 50 Hz 轨迹取 1/4，12.5 fps 的 GIF 足够观察步态
  data.qpos[:] = q
  mujoco.mj_forward(model, data)
  renderer.update_scene(data, camera)
  frames.append(renderer.render().copy())
renderer.close()

imageio.mimsave("rollout.gif", frames, fps=13)
```

完整的一键交互版本见仓库内的 `notebooks/go2_training_demo.ipynb`。

---

上一步：[← 1. 环境安装与硬件约束](./environment-setup.md) · 下一步：[3. ONNX 导出与官方栈 Sim2Sim →](./sim2sim.md)
