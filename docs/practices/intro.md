---
title: 项目实战概述
sidebar_position: 1
---

# 项目实战：围绕主流本体做可展示项目

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

- [玩转CS123机器人控制](./amd/cs123/intro)：面向 AMD 平台的 CS123 机器人控制项目入口。
