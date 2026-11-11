---
title: "轮足系列"
description: "轮足机器人项目总览：两轮足 Flamingo 在 Isaac Lab 中的欠驱动平衡、PPO 与 CaT 约束式强化学习及 MuJoCo 跨仿真迁移。"
sidebar_position: 1
displayed_sidebar: practicesWheelLeggedSidebar
---

# 轮足系列

围绕兼顾**轮式高效率移动**与**足式越障适应性**的轮足复合机器人，探索欠驱动动态平衡、约束式强化学习（Constraints as Termination, CaT）以及从 **Isaac Lab** 到 **MuJoCo** 的跨物理引擎零样本迁移（Sim2Sim）。

## 🗂️ 项目总览 \{#projects}

| 分类 | 项目 | 简介 |
| :--- | :--- | :--- |
| 工业栈实战课程 | [两轮足 Flamingo · Isaac Lab](./flamingo-isaaclab/preview) | 基于 8 自由度两轮足机器人 Flamingo，打通 Isaac Lab 并行训练、PPO vs CaT 对比、ONNX 策略导出与 MuJoCo 抗扰动零样本验证（课程预告） |

## 🎬 典型演示 \{#demos}

| [![多个 Flamingo 两轮足机器人在 Isaac Lab 中并行仿真](./flamingo-isaaclab/figs/hero_swarm.gif)](./flamingo-isaaclab/preview)<br/>**[两轮足 Flamingo · Isaac Lab 大规模并行强化学习](./flamingo-isaaclab/preview)**<br/>多机器人同时采样随机线速度与转向角速度指令 · 轮式倒立摆欠驱动动态平衡与转弯控制 |
| :---: |

## 🧭 内容大纲 \{#outline}

### 两轮足 Flamingo · Isaac Lab 训练与跨仿真迁移 \{#flamingo}

相比静态稳定的四足入门平台，8 自由度两轮足兼具轮式倒立摆欠驱动特性、较低显存开销与高工业契合度，适合作为掌握了基础控制与单仿真 RL 之后的第二个进阶项目。

| 规划章节 | 内容 |
| :--- | :--- |
| [0. 课程预告与技术选型](./flamingo-isaaclab/preview) | 两轮足欠驱动特性、与 CS123 四足课程的技术栈对照、硬件/软件前置要求（Isaac Sim 4.5 + Isaac Lab） |
| 章 1：本体与环境（规划中） | Flamingo 8-DoF 腿轮构型、URDF / USD 资产准备与 Isaac Lab 任务环境注册 |
| 章 2：奖励设计与 PPO 训练（规划中） | 线速度/角速度跟踪、机身直立姿态稳定项与 `rsl_rl` 大规模并行 PPO 训练 |
| 章 3：约束式强化学习 CaT（规划中） | 硬约束终止与软奖励塑形的取舍，以及 CaT 与标准 PPO 的收敛曲线和行为对比 |
| 章 4：策略导出与 Sim2Sim（规划中） | ONNX 策略导出、关节顺序/PD 增益对齐与 MuJoCo 端零样本推理回放 |
| 章 5–6：鲁棒性评测与项目交付（规划中） | 横向推力、初始姿态扰动、地面摩擦随机化测试与工程产出物梳理 |
