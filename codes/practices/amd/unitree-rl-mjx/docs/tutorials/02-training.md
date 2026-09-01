# 教程 2：训练 Go2 速度策略

本章训练 Go2 平地速度跟踪任务：策略接收本体感知观测与三维速度指令
（前向 vx、侧向 vy、偏航角速度 wz），输出 12 个关节的位置目标。
奖励与课程设计移植自 Unitree 官方的 `unitree_rl_mjlab`（Isaac Lab 生态），
物理后端换成 MuJoCo MJX，因此整个训练循环在单张 GPU 上以 JAX 编译执行。

## 冒烟训练（约 20 分钟）

先用 4000 万步确认整条链路（ROCm 机器记得带上教程 1 的 XLA_FLAGS）：

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  uv run python -m unitree_rl_mjx.train.go2_velocity \
  --seed 0 --num-timesteps 41943040 --out-dir runs/go2-smoke
```

训练过程中每个评估点打印一行 `steps reward`；奖励应从负值起步、
数百万步内转正并持续上升（4000 万步时约 40+）。

## 完整训练

不带 `--num-timesteps` 即训练完整的 9.8 亿步公开预算：

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  uv run python -m unitree_rl_mjx.train.go2_velocity \
  --seed 0 --out-dir runs/go2-full
```

实测墙钟：Radeon AI PRO R9700 约 7.9 小时，NVIDIA A40 约 5.5 小时；
两个后端训练出的策略经留出指标验证统计不可区分。
训练在约 4.9 亿步处会经历一次指令课程加宽（回报短暂回落后恢复），属预期现象。

## 输出文件

`--out-dir` 下会得到：

| 文件 | 内容 |
|---|---|
| `metrics.jsonl` | 逐评估点的回报与训练吞吐（JSON lines） |
| `params.bin` | 训练完的策略参数（brax 格式，教程 3 用它导出 ONNX） |
| `trajectory.npz` | 训练结束后一条评估轨迹的 qpos 序列 |
| `run.json` | 运行清单：后端、设备、版本、墙钟、随机种子 |

## 画训练曲线

```python
import json
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("runs/go2-smoke/metrics.jsonl")]
steps = [r["step"] for r in rows if "eval/episode_reward" in r]
reward = [r["eval/episode_reward"] for r in rows if "eval/episode_reward" in r]
plt.plot(steps, reward)
plt.xlabel("environment steps")
plt.ylabel("episode reward")
plt.show()
```

## 渲染训练结果

`trajectory.npz` 配合任务自带的场景模型即可离线渲染：

```python
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
for q in qpos[::4]:  # 50 Hz 轨迹取 1/4，12.5 fps 的 gif 足够看步态
  data.qpos[:] = q
  mujoco.mj_forward(model, data)
  renderer.update_scene(data, camera)
  frames.append(renderer.render().copy())
renderer.close()

import imageio
imageio.mimsave("rollout.gif", frames, fps=13)
```

完整的一键版本见 `notebooks/go2_training_demo.ipynb`。

下一步：[教程 3：导出策略并在官方仿真栈闭环](03-sim2sim.md)
