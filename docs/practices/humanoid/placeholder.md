---
title: "双足系列"
description: "双足、轮腿与人形机器人项目总览：两轮足 Flamingo（Isaac Lab 完整 6 章课程 + CaT + MuJoCo Sim2Sim）、MicroDuck RL 平地行走/起身/踢球、12 步跳墙越障前滚翻课程与 AMD ROCm 并行训练。"
sidebar_position: 1
displayed_sidebar: practicesHumanoidSidebar
---

# 双足系列

围绕双足、轮腿双足与人形机器人的高动态欠驱动平衡、强化学习与跨仿真验证，涵盖两条完整动手主线：一条是基于 **NVIDIA Isaac Lab + 约束式强化学习（CaT）+ ONNX + MuJoCo Sim2Sim** 的 **8 自由度两轮足 Flamingo 6 章完整课程**；另一条是从交互式奖励实验进阶到**跑—跳—越墙—前滚翻—起身 12 步长链课程**的 **MicroDuck RL**。

## 🗂️ 项目总览 \{#projects}

| 分类 | 项目 | 简介 |
| :--- | :--- | :--- |
| 全尺寸人形（4 篇） | [Unitree G1 人形运控实战](./unitree-g1/) | 从 16 项奖励深挖、10,000 轮 PPO 曲线、0.2–0.8 m/s 多速度 MuJoCo Sim2Sim 与 [浏览器 LSTM 仿真](/simulation)，延伸至 AMP、Get-Up 与 GR00T WBC |
| 轮足完整课程（6 章） | [两轮足 Flamingo · Isaac Lab](./flamingo-isaaclab/intro) | 围绕 8 自由度双足轮腿机器人，从延迟/齿轮执行器建模、4096 环境并行 PPO、约束式强化学习（CaT），一路打通 ONNX 融合导出与 MuJoCo 零样本抗扰动压测 |
| 入门动手主线 | [MicroDuck RL：教小黄鸭学会一个动作](./microduck-rl/) | 通过网页内动作回放与奖励设计互动实验建立直觉，再用 `mjlab` + MuJoCo Warp 启动第一次双足 PPO 训练 |
| 进阶课程学习 | [MicroDuck 跳墙课程：12 步复现跑跳滚翻全链](./microduck-rl/wall-curriculum) | 用 12 步课程串联助跑、起跳、越过 5 cm 矮墙、前滚翻与起身继续跑，拆解镜像对称损失、防套利奖励与反向课程破局 |
| 评测与探索 | [MicroDuck：实验记录与创意动作](./microduck-rl/experiment-notes) | 轮式起身、轮式转向、踢球收势与多模式固定条件离屏验收数据及创意动作探索 |
| AMD 专区旗舰 | [MicroDuck RL｜AMD ROCm](/docs/practices/amd/microduck-rl) | 在 Radeon AI PRO R9700 上从源码编译 ROCm Warp / MuJoCo Warp，修复动态接触缓存并完成双足 PPO 长训 |
| 进阶专题（3 篇） | [双足动态平衡](./balance-control/placeholder) / [全身运动跟踪](./motion-tracking/placeholder) / [双足任务规划](./task-planning-demo/placeholder) | 双足抗扰动平衡与外力推撞恢复（含 [Unitree G1 在线仿真](/simulation)）、全身动捕重定向跟踪与高层技能编排 |

## 🎬 典型演示 \{#demos}

### Unitree G1 人形运控与两轮足 Flamingo 并行训练 \{#flamingo-demos}

| [![Unitree G1 多速度 MuJoCo Sim2Sim 对比](./unitree-g1/figs/g1_sim2sim_speed_comparison.gif)](./unitree-g1/)<br/>**[Unitree G1 人形运控实战（4 篇）](./unitree-g1/)**<br/>16 项奖励深挖 · 10,000 轮 PPO · 多速度 Sim2Sim · 浏览器 LSTM 实时推撞 | [![多个 Flamingo 两轮足机器人在 Isaac Lab 中并行仿真](./flamingo-isaaclab/figs/hero_swarm.gif)](./flamingo-isaaclab/intro)<br/>**[两轮足 Flamingo · Isaac Lab 6 章完整实战课程](./flamingo-isaaclab/intro)**<br/>8 自由度轮腿欠驱动动态平衡 · PPO vs CaT 约束式强化学习 · MuJoCo 抗扰压测 |
| :---: | :---: |

### MicroDuck 基础动作与固定条件验收 \{#microduck-demos}

| [![MicroDuck 稳定行走与轮式转向](./microduck-rl/figs/focus-final/roller_swizzle_final.gif)](./microduck-rl/)<br/>**[MicroDuck 稳定行走与轮式转向](./microduck-rl/)**<br/>mjlab + MuJoCo Warp · 固定条件零倒地帧回放 | [![MicroDuck 趴地自主起身](./microduck-rl/figs/focus-final/roller_standup_final.gif)](./microduck-rl/experiment-notes)<br/>**[MicroDuck 趴地自主起身](./microduck-rl/experiment-notes)**<br/>RollerStandUp · 从俯卧状态发力起身并恢复直立平衡 | [![MicroDuck 踢球与收势平衡](./microduck-rl/figs/focus-final/ballkick_final.gif)](./microduck-rl/)<br/>**[MicroDuck 踢球与收势平衡](./microduck-rl/)**<br/>BallKick-Flat · 支撑腿稳姿、摆腿击球并稳定收势 |
| :---: | :---: | :---: |

### 12 步跳墙越障课程与 AMD ROCm 并行训练 \{#curriculum-demos}

| [![MicroDuck 平地前滚翻基础技能](./microduck-rl/figs/wall-curriculum/Mjlab-Roulade-Flat-MicroDuck.gif)](./microduck-rl/wall-curriculum)<br/>**[第 0 步：平地前滚翻（Roulade）](./microduck-rl/wall-curriculum)**<br/>课程前传基础技能 · 翻滚角动量与抱膝姿态衔接 | [![MicroDuck 5cm 越墙跑跳滚翻全链](./microduck-rl/figs/wall-curriculum/b22_5cm_v2.gif)](./microduck-rl/wall-curriculum)<br/>**[第 12 步：5 cm 越墙跑跳滚翻全链](./microduck-rl/wall-curriculum)**<br/>助跑 → 起跳飞越矮墙 → 前滚翻缓冲 → 起身继续奔跑 | [![MicroDuck AMD ROCm R9700 实测步态](../amd/microduck-rl/figs/microduck-amd-r9700-model750.gif)](/docs/practices/amd/microduck-rl)<br/>**[MicroDuck RL · AMD ROCm 实测](/docs/practices/amd/microduck-rl)**<br/>Radeon R9700 · ROCm MuJoCo Warp 并行 PPO 步态 |
| :---: | :---: | :---: |

## 🧭 内容大纲 \{#outline}

### 0. Unitree G1 人形运控实战（4 篇完整专题） \{#unitree-g1-course}

围绕 Unitree G1（12-DoF 下肢行走主线 + 29-DoF 全身控制延伸），串联 16 项奖励拆解、万轮训练诊断、多速度 MuJoCo Sim2Sim、浏览器无状态 LSTM 部署与 AMP / Get-Up / GR00T WBC 进阶经验。

| 篇章 | 核心内容 |
| :--- | :--- |
| [Unitree G1 人形运控总览](./unitree-g1/) | 12-DoF / 29-DoF 规格、47 维观测契约、万轮基线指标与 [在线仿真](/simulation) 入口 |
| [1. 奖励函数与训练动力学](./unitree-g1/reward-and-training) | 16 项奖励 6 大分组、公式与权重尺度对照、先存活后提速曲线与 8,000 轮后再平衡归因 |
| [2. Sim2Sim 与浏览器 LSTM 部署](./unitree-g1/sim2sim-and-web) | 单进程/DDS 两级 Sim2Sim、0.2/0.5/0.8 m/s 对比回放、无状态 LSTM ONNX 与 24 MB → 1.49 MB 网格轻量化 |
| [3. AMP / 起身与 GR00T 全身控制](./unitree-g1/whole-body-extensions) | AMP 复现与 RSI 瞬移假象避坑铁律、HoST/HumanUP 防空翻起身配方、SONIC Latent Action 架构 |

### 1. 两轮足 Flamingo · Isaac Lab（6 章完整课程） \{#flamingo-course}

围绕轮式倒立摆欠驱动平衡与轮腿异构控制，完整复刻工业界主流“Isaac Lab 大规模训练 → 约束式 RL 优化 → ONNX 部署图固化 → MuJoCo 跨引擎压测”工作流。

| 章节 | 核心内容 |
| :--- | :--- |
| [0. 课程总览](./flamingo-isaaclab/intro) | 两轮足欠驱动特性、全链路架构图、双环境隔离配置（Isaac Sim 4.5 + Isaac Lab 2.1 / MuJoCo） |
| [1. 本体建模与环境注册](./flamingo-isaaclab/robot-and-env) | 8-DoF 轮腿拓扑、膝关节同步带减速比 `-1.5`、`0~20 ms` 延迟执行器、混合动作与非对称堆叠观测 |
| [2. 奖励设计与 PPO 训练](./flamingo-isaaclab/reward-and-ppo) | 非最小相位起步分析、16 项奖励拆解、三层域随机化、4096 环境并行 PPO 与四类病态步态排障 |
| [3. 约束式强化学习（CaT）](./flamingo-isaaclab/cat-constrained-rl) | Constraints as Terminations 虚拟随机终止与 GAE 连续折现推导、`ConstraintManager` 源码与 PPO 对照 |
| [4. 策略导出与 MuJoCo Sim2Sim](./flamingo-isaaclab/onnx-and-sim2sim) | `Normalizer + Actor` 融合导出 ONNX、攻克 DFS/BFS 关节序与四元数五大跨引擎陷阱、零样本闭环 |
| [5. 鲁棒性与抗扰动评测](./flamingo-isaaclab/robustness-eval) | 矢状面/冠状面速度脉冲、低摩擦路面、转动惯量三角不等式配平与 5 项量化指标矩阵 |
| [6. 项目复盘与工程交付](./flamingo-isaaclab/delivery-and-interview) | 标准化交付档案清单、四大架构权衡复盘、后空翻/台阶进阶路线与 5 道高频面试深度问答 |

### 2. MicroDuck RL 基础与奖励工程 \{#microduck-basics}

不需要先推完所有强化学习公式：先在网页端通过交互式实验理解观测、动作、奖励权重与套利行为（Reward Hacking），再在本地 GPU 环境跑通完整训练与离屏渲染。

| 篇章 | 内容 |
| :--- | :--- |
| [MicroDuck RL：教小黄鸭学会一个动作](./microduck-rl/) | 动作回放画廊、交互式奖励实验室、`Mjlab-Velocity-Flat-MicroDuck` 并行训练与 checkpoint 回放 |
| [MicroDuck：实验记录与创意动作](./microduck-rl/experiment-notes) | 固定 seed 离屏验收标准、`evaluation.tsv` 量化筛查与跨模式训练记录复盘 |

### 3. 12 步跳墙课程：复杂多阶段动作链设计 \{#wall-curriculum}

展示如何把单一运动技能扩展为“助跑—起跳—越障—前滚翻—起身续跑”的高动态长时程动作链，包含真实踩坑记录与破局方法。

| 阶段 | 核心内容 |
| :--- | :--- |
| [前传与起跳（第 0–5 步）](./microduck-rl/wall-curriculum) | 平地翻滚预训练、无墙跳跃引导、1 cm / 2 cm 真实矮墙引入与推墙/趴爬防套利门控 |
| [反向课程破局（第 6–9 步）](./microduck-rl/wall-curriculum) | 突破 180°–220° 仰面势阱的 midroll 反向课程初始化，打通翻滚后起身继续奔跑全链 |
| [墙高递进验收（第 10–12 步）](./microduck-rl/wall-curriculum) | 保持奖励结构不变，沿 `0.5 cm → 1 cm → 2 cm → 3.5 cm → 5 cm` 逐档升墙并通过 GIF 验收 |

### 4. AMD ROCm 平台适配与后续进阶方向 \{#roadmap}

| 篇章 | 内容 |
| :--- | :--- |
| [MicroDuck RL｜AMD ROCm](/docs/practices/amd/microduck-rl) | ROCm Warp / MuJoCo Warp 源码构建、动态 broadphase 接触缓存修复、graph capture 回归与 R9700 训练评测 |
| [双足动态平衡与抗扰控制](./balance-control/placeholder) | 质心/角动量与 Capture Point 平衡控制、外力推撞测试及 [Unitree G1 在线仿真](/simulation) |
| [人形全身运动跟踪](./motion-tracking/placeholder) | 动作捕捉参考轨迹重定向、关键点追踪奖励设计与全身模仿策略训练 |
| [双足高层任务规划](./task-planning-demo/placeholder) | 高层多模态指令分解、状态机/行为树编排与底层双足运动技能安全衔接 |
