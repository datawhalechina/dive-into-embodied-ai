---
title: 项目实战概述
sidebar_position: 1
displayed_sidebar: practicesOverviewSidebar
---

# 项目实战：从仿真验证走向真机部署

选择一个具体任务，通过分章项目、Demo 或复现案例跑通环境、训练、控制与评估。既可以沿着章节逐步搭建完整系统，也可以选择一个独立实验验证方法。遇到原理上的疑问，到 [理论基础](/docs/foundations/intro) 查找对应模块；查找论文、数据和代码来源，可以进入具身导论的 [领域资源](/docs/introduction/resources)。

## 从哪里开始 \{#getting-started}

- 想从零搭建一个系统：从 [四足机器人项目](./quadruped/cs123/intro) 开始，按 8 章顺序学习执行器、运动学、建模、步态与策略训练。需要 Python 基础，控制和线性代数可以结合实验补齐。
- 想学习机械臂数据与真机操作：先读 [LeRobot 中文课程讲义](./robot-arm/data-collection/lerobot-course)，再进入 [SO-101 真机教程](./robot-arm/data-collection/so101-lerobot-real)。讲义已整理 Unit 0–2，需要 Python 基础；后续操作需要对应硬件。
- 想先体验一个仿真效果：选择 [MicroDuck RL](./humanoid/microduck-rl/) 或 [ACT 双臂操作](./vla/act)，按项目要求准备环境。
- 想进一步学习轮足平衡控制：查看 [Flamingo 项目预告](./wheel-legged/flamingo-isaaclab/preview)，了解前置知识和计划中的实验。

动手前先确认环境和目标，按步骤保留代码、参数与运行结果，再用评测和失败案例检查自己的理解。

## 方向一览 \{#directions}

| 方向 | 项目数 | 适合 |
|------|--------|------|
| [机械臂](./robot-arm/placeholder) | 5 个 | VLA / 数据采集 / 模仿学习入门 |
| [四足机器人](./quadruped/placeholder) | 4 个 | 强化学习 / 控制 / sim2real |
| [双足 / 人形](./humanoid/placeholder) | 1 个已上线，2 个规划中 | 进阶控制 / 强化学习 / 任务规划 |
| [移动操作](./mobile-manipulation/placeholder) | 3 个 | 导航 + 操作综合 |
| [轮足机器人](./wheel-legged/placeholder) | 预告 | 欠驱动平衡 / Isaac Lab / 跨仿真验证 |

## 已上线项目 \{#available-projects}

- [MicroDuck RL：让小黄鸭学会走路、起身和翻滚](./humanoid/microduck-rl/)：从 MJCF、BAM 执行器和 mjlab 任务注册开始，在 MuJoCo Warp 中训练 18 个动作模式，并用 GIF / MP4 展示离屏评估结果。

## AMD 专区 \{#amd}

- [AUP Learning Cloud 云算力](./amd/aup-learning-cloud)：在浏览器中使用 Ryzen AI APU、JupyterHub、Code Server 与 ROCm 环境，适合课程练习、端侧推理和小规模实验。
- [MicroDuck RL｜AMD ROCm](./amd/microduck-rl)：在 Radeon R9700 上编译 ROCm Warp / MuJoCo Warp，修复动态 broadphase 缓存并训练双足 PPO。
- [ACT 双臂操作训练｜AMD ROCm](./amd/vla-act)：在 Radeon GPU 上完成 ACT BF16 训练、断点续训、20 回合评测与视频导出。
- [玩转 Pupper 四足机器人](./amd/pupper-control/intro)：AMD 专区旗舰项目，包含 **Pupper Locomotion｜强化学习运动策略**与 **Pupper VLA｜视觉-语言-动作智能**两个方向。

## 仿真实战 \{#simulation}

| 项目 | 技术主线 | 状态 |
| --- | --- | --- |
| [从零到一搭建四足机器人](./quadruped/cs123/intro) | MuJoCo、PD 控制、运动学、步态、策略训练与感知 | 总览与 8 章可读 |
| [MicroDuck RL：走路、起身和翻滚](./humanoid/microduck-rl/) | mjlab、MuJoCo Warp、CUDA 并行 PPO、18 个动作模式 | 可用 |
| [MuJoCo 仿真入门](./robot-arm/mujoco-arm-pick-place) | MJCF、物理仿真、Python 控制 | 可用 |
| [DDPG InvertedPendulum](./robot-arm/ddpg-mujoco/invertedpendulum-v5) | 连续控制基础与 DDPG baseline | 可用 |
| [DDPG Reacher](./robot-arm/ddpg-mujoco/reacher-v5) | 二维机械臂目标追踪 | 可用 |
| [DDPG Pusher](./robot-arm/ddpg-mujoco/pusher-v5) | 机械臂接触操作与奖励设计 | 可用 |
| [ACT 双臂操作训练](./vla/act) | ALOHA、模仿学习、ACT、多回合评估 | 可用 |
| [π₀.₅ + RECAP：LIBERO 复现](./vla/recap) | 价值模型、优势标注、ACP、LIBERO 评估 | 可用 |
| [Sim2Sim 验证](./quadruped/sim2sim/placeholder) | 跨仿真策略验证 | 施工中 |
| [两轮足 Flamingo · Isaac Lab](./wheel-legged/flamingo-isaaclab/preview) | 轮足平衡、强化学习训练与跨仿真验证 | 课程预告 |

## 真机实战 \{#real-robots}

LeRobot 讲义提供数据与工具链准备，SO-101 项目再将这些知识接到真实机械臂上。

| 项目 | 技术主线 | 状态 |
| --- | --- | --- |
| [LeRobot 中文课程讲义](./robot-arm/data-collection/lerobot-course) | 机器人学习、数据集、工具链与经典机器人学 | 已整理 Unit 0–2 |
| [SO-101 + LeRobot 真机教程](./robot-arm/data-collection/so101-lerobot-real) | 硬件连通、安全测试、动作回放 | 可用 |
| [ROS2 机械臂控制](./robot-arm/ros2-arm-control/placeholder) | ROS2 控制链路与机械臂执行 | 施工中 |
| [Sim2Real 指南](./quadruped/sim2real-guide/placeholder) | 仿真策略部署与真机验证 | 施工中 |
