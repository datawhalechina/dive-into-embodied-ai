---
title: 项目实战概述
sidebar_position: 1
---

# 项目实战：从仿真验证走向真机部署

项目实战按企业关心的本体和任务组织，每个项目统一模板：项目背景 → 岗位映射 → 技术路线 → 代码实战 → 实验结果 → 简历写法 → 面试追问。

## 方向一览

| 方向 | 项目数 | 适合 |
|------|--------|------|
| [机械臂](./robot-arm/placeholder) | 5 个 | VLA / 数据采集 / 模仿学习入门 |
| [四足机器人](./quadruped/placeholder) | 4 个 | 强化学习 / 控制 / sim2real |
| [双足 / 人形](./humanoid/placeholder) | 1 个已上线，2 个规划中 | 进阶控制 / 强化学习 / 任务规划 |
| [移动操作](./mobile-manipulation/placeholder) | 3 个 | 导航 + 操作综合 |
| [轮足机器人](./wheel-legged/placeholder) | 预告 | 欠驱动平衡 / Isaac Lab / 跨仿真验证 |

## 独立课程

- [从零到一搭建四足机器人](./quadruped/cs123/intro)：CS123 四足仿真课程，按 8 章独立组织。
- [LeRobot 中文课程讲义](./robot-arm/data-collection/lerobot-course)：机器人学习与 LeRobot 工具链主线。
- [SO-101 + LeRobot 真机教程](./robot-arm/data-collection/so101-lerobot-real)：从硬件连接到策略回放的最小真机流程。
- [两轮足 Flamingo · Isaac Lab](./wheel-legged/flamingo-isaaclab/preview)：Isaac Lab + rsl_rl 训练两轮足策略，导出 ONNX 后在 MuJoCo 中零样本验证。

## 已上线项目

- [MicroDuck RL：双足机器人 GPU 仿真](./humanoid/microduck-rl/)：从 MJCF、BAM 执行器和 mjlab 任务注册开始，在 MuJoCo Warp 中完成 GPU 并行 PPO smoke train，并导出 ONNX 推理产物。

## AMD 专区

- [AUP Learning Cloud 云算力](./amd/aup-learning-cloud)：在浏览器中使用 Ryzen AI APU、JupyterHub、Code Server 与 ROCm 环境，适合课程练习、端侧推理和小规模实验。
- [ACT 双臂操作训练｜AMD ROCm](./amd/vla-act)：在 Radeon GPU 上完成 ACT BF16 训练、断点续训、20 回合评测与视频导出。
- [玩转 Pupper 四足机器人](./amd/pupper-control/intro)：AMD 专区旗舰项目，包含 **Pupper Locomotion｜强化学习运动策略**与 **Pupper VLA｜视觉-语言-动作智能**两个方向。

## 仿真实战

| 项目 | 技术主线 | 状态 |
| --- | --- | --- |
| [从零到一搭建四足机器人](./quadruped/cs123/intro) | MuJoCo、PD、运动学、PPO、LLM 控制 | 可用 |
| [MicroDuck RL 小黄鸭双足机器人](./humanoid/microduck-rl/) | mjlab、MuJoCo Warp、GPU 并行 PPO、双足步态 | 可用 |
| [MuJoCo 仿真入门](./robot-arm/mujoco-arm-pick-place) | MJCF、物理仿真、Python 控制 | 可用 |
| [DDPG InvertedPendulum](./robot-arm/ddpg-mujoco/invertedpendulum-v5) | 连续控制基础与 DDPG baseline | 可用 |
| [DDPG Reacher](./robot-arm/ddpg-mujoco/reacher-v5) | 二维机械臂目标追踪 | 可用 |
| [DDPG Pusher](./robot-arm/ddpg-mujoco/pusher-v5) | 机械臂接触操作与奖励设计 | 可用 |
| [ACT 双臂操作训练](./vla/act) | ALOHA、模仿学习、ACT、多回合评估 | 可用 |
| [两轮足 Flamingo · Isaac Lab](./wheel-legged/flamingo-isaaclab/preview) | PPO / CaT、Sim2Sim、鲁棒性验证 | 预告 |
| [Sim2Sim 验证](./quadruped/sim2sim/placeholder) | 跨仿真策略验证 | 施工中 |

## 真机实战

| 项目 | 技术主线 | 状态 |
| --- | --- | --- |
| [SO-101 + LeRobot 真机教程](./robot-arm/data-collection/so101-lerobot-real) | 硬件连通、安全测试、动作回放 | 可用 |
| [LeRobot 中文课程讲义](./robot-arm/data-collection/lerobot-course) | 数据集、机器人学习工具链、真机流程前置知识 | 可用 |
| [ROS2 机械臂控制](./robot-arm/ros2-arm-control/placeholder) | ROS2 控制链路与机械臂执行 | 施工中 |
| [Sim2Real 指南](./quadruped/sim2real-guide/placeholder) | 仿真策略部署与真机验证 | 施工中 |
