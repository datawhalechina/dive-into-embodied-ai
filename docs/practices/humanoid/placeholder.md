---
title: "双足系列"
description: "双足与人形机器人项目总览：MicroDuck RL 平地行走/起身/踢球、12 步跳墙越障前滚翻课程与 AMD ROCm 并行训练。"
sidebar_position: 1
displayed_sidebar: practicesHumanoidSidebar
---

# 双足系列

围绕双足与人形机器人的高动态平衡、强化学习与课程学习设计，从**交互式奖励函数实验与首次 GPU 并行 PPO 训练**出发，进阶到**跑—跳—越墙—前滚翻—起身 12 步长链课程复现**及跨平台加速实测。

## 🗂️ 项目总览 \{#projects}

| 分类 | 项目 | 简介 |
| :--- | :--- | :--- |
| 入门动手主线 | [MicroDuck RL：教小黄鸭学会一个动作](./microduck-rl/) | 通过网页内动作回放与奖励设计互动实验建立直觉，再用 `mjlab` + MuJoCo Warp 启动第一次双足 PPO 训练 |
| 进阶课程学习 | [MicroDuck 跳墙课程：12 步复现跑跳滚翻全链](./microduck-rl/wall-curriculum) | 用 12 步课程串联助跑、起跳、越过 5 cm 矮墙、前滚翻与起身继续跑，拆解镜像对称损失、防套利奖励与反向课程破局 |
| 评测与探索 | [MicroDuck：实验记录与创意动作](./microduck-rl/experiment-notes) | 轮式起身、轮式转向、踢球收势与多模式固定条件离屏验收数据及创意动作探索 |
| AMD 专区旗舰 | [MicroDuck RL｜AMD ROCm](/docs/practices/amd/microduck-rl) | 在 Radeon AI PRO R9700 上从源码编译 ROCm Warp / MuJoCo Warp，修复动态接触缓存并完成双足 PPO 长训 |
| 规划章节 | [平衡控制](./balance-control/placeholder) / [运动跟踪](./motion-tracking/placeholder) / [任务规划 Demo](./task-planning-demo/placeholder) | 双足抗扰动平衡、全身动作重定向跟踪（Motion Tracking）与高层任务规划（施工中） |

## 🎬 典型演示 \{#demos}

### MicroDuck 基础动作与固定条件验收 \{#microduck-demos}

| [![MicroDuck 稳定行走与轮式转向](./microduck-rl/figs/focus-final/roller_swizzle_final.gif)](./microduck-rl/)<br/>**[MicroDuck 稳定行走与轮式转向](./microduck-rl/)**<br/>mjlab + MuJoCo Warp · 固定条件零倒地帧回放 | [![MicroDuck 趴地自主起身](./microduck-rl/figs/focus-final/roller_standup_final.gif)](./microduck-rl/experiment-notes)<br/>**[MicroDuck 趴地自主起身](./microduck-rl/experiment-notes)**<br/>RollerStandUp · 从俯卧状态发力起身并恢复直立平衡 | [![MicroDuck 踢球与收势平衡](./microduck-rl/figs/focus-final/ballkick_final.gif)](./microduck-rl/)<br/>**[MicroDuck 踢球与收势平衡](./microduck-rl/)**<br/>BallKick-Flat · 支撑腿稳姿、摆腿击球并稳定收势 |
| :---: | :---: | :---: |

### 12 步跳墙越障课程与 AMD ROCm 并行训练 \{#curriculum-demos}

| [![MicroDuck 平地前滚翻基础技能](./microduck-rl/figs/wall-curriculum/Mjlab-Roulade-Flat-MicroDuck.gif)](./microduck-rl/wall-curriculum)<br/>**[第 0 步：平地前滚翻（Roulade）](./microduck-rl/wall-curriculum)**<br/>课程前传基础技能 · 翻滚角动量与抱膝姿态衔接 | [![MicroDuck 5cm 越墙跑跳滚翻全链](./microduck-rl/figs/wall-curriculum/b22_5cm_v2.gif)](./microduck-rl/wall-curriculum)<br/>**[第 12 步：5 cm 越墙跑跳滚翻全链](./microduck-rl/wall-curriculum)**<br/>助跑 → 起跳飞越矮墙 → 前滚翻缓冲 → 起身继续奔跑 | [![MicroDuck AMD ROCm R9700 实测步态](../amd/microduck-rl/figs/microduck-amd-r9700-model750.gif)](/docs/practices/amd/microduck-rl)<br/>**[MicroDuck RL · AMD ROCm 实测](/docs/practices/amd/microduck-rl)**<br/>Radeon R9700 · ROCm MuJoCo Warp 并行 PPO 步态 |
| :---: | :---: | :---: |

## 🧭 内容大纲 \{#outline}

### 1. MicroDuck RL 基础与奖励工程 \{#microduck-basics}

不需要先推完所有强化学习公式：先在网页端通过交互式实验理解观测、动作、奖励权重与套利行为（Reward Hacking），再在本地 GPU 环境跑通完整训练与离屏渲染。

| 篇章 | 内容 |
| :--- | :--- |
| [MicroDuck RL：教小黄鸭学会一个动作](./microduck-rl/) | 动作回放画廊、交互式奖励实验室、`Mjlab-Velocity-Flat-MicroDuck` 并行训练与 checkpoint 回放 |
| [MicroDuck：实验记录与创意动作](./microduck-rl/experiment-notes) | 固定 seed 离屏验收标准、`evaluation.tsv` 量化筛查与跨模式训练记录复盘 |

### 2. 12 步跳墙课程：复杂多阶段动作链设计 \{#wall-curriculum}

展示如何把单一运动技能扩展为“助跑—起跳—越障—前滚翻—起身续跑”的高动态长时程动作链，包含真实踩坑记录与破局方法。

| 阶段 | 核心内容 |
| :--- | :--- |
| [前传与起跳（第 0–5 步）](./microduck-rl/wall-curriculum) | 平地翻滚预训练、无墙跳跃引导、1 cm / 2 cm 真实矮墙引入与推墙/趴爬防套利门控 |
| [反向课程破局（第 6–9 步）](./microduck-rl/wall-curriculum) | 突破 180°–220° 仰面势阱的 midroll 反向课程初始化，打通翻滚后起身继续奔跑全链 |
| [墙高递进验收（第 10–12 步）](./microduck-rl/wall-curriculum) | 保持奖励结构不变，沿 `0.5 cm → 1 cm → 2 cm → 3.5 cm → 5 cm` 逐档升墙并通过 GIF 验收 |

### 3. AMD ROCm 平台适配与后续进阶方向 \{#roadmap}

| 篇章 | 内容 |
| :--- | :--- |
| [MicroDuck RL｜AMD ROCm](/docs/practices/amd/microduck-rl) | ROCm Warp / MuJoCo Warp 源码构建、动态 broadphase 接触缓存修复、graph capture 回归与 R9700 训练评测 |
| [平衡控制](./balance-control/placeholder) | 双足推力扰动恢复与质心/角动量平衡控制（施工中） |
| [运动跟踪](./motion-tracking/placeholder) | 动作捕捉参考轨迹重定向与全身模仿跟踪（施工中） |
| [任务规划 Demo](./task-planning-demo/placeholder) | 高层多模态指令分解与底层双足运动技能衔接（施工中） |
