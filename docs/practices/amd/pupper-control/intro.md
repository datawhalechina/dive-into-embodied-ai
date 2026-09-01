---
title: 玩转 Pupper 四足机器人
sidebar_position: 1
displayed_sidebar: practicesAmdSidebar
description: "AMD 专区旗舰项目：围绕 Pupper 完成强化学习运动策略与视觉-语言-动作智能实验。"
---

# 玩转 Pupper 四足机器人

这是 **AMD 专区的旗舰项目**。项目围绕同一台 Pupper 四足机器人，组织两条可以独立完成、也可以前后衔接的实验主线：先训练稳定可用的运动策略，再把视觉与自然语言指令接入机器人的动作能力。

`AMD`　`Pupper`　`RL Locomotion`　`VLA`

## 两个实验方向

| 方向 | 核心问题 | 主要产出 |
|---|---|---|
| [Pupper Locomotion｜强化学习运动策略](./locomotion) | 如何让 Pupper 学会稳定地跟踪速度指令 | PPO 策略、训练曲线、速度跟踪与鲁棒性实验 |
| [Pupper VLA｜视觉-语言-动作智能](./vla) | 如何让 Pupper 根据视觉信息和语言指令自主执行任务 | VLA Pipeline、技能接口、端到端任务 Demo |

### Pupper Locomotion｜强化学习运动策略

基于 MuJoCo、Gymnasium 和 PPO 组织完整的运动策略实验，重点覆盖 Observation 与 Action 设计、奖励设计、速度指令跟踪、Domain Randomization，以及 AMD 平台上的训练与验证。

[进入 Pupper Locomotion](./locomotion)

### Pupper VLA｜视觉-语言-动作智能

把视觉感知、自然语言理解和机器人运动技能连接起来，让 Pupper 能够理解环境与任务，并调用合适的动作完成目标。方向以“找到红色的球并走过去”为代表性 Demo。

[进入 Pupper VLA](./vla)

## 内容边界与前置教程

本项目聚焦 AMD 平台上的训练、推理与完整实验，不重复讲解 **PD 控制、正逆运动学、Pupper 建模与搭建**。

如果还没有完成这些基础内容，请先学习 [《从零到一搭建四足机器人》](/docs/practices/quadruped/cs123/intro)。后续实验直接复用该教程中的 Pupper 仿真模型、底层控制接口、视觉感知和语言控制能力。

## 推荐学习顺序

1. 完成基础教程，确认 Pupper 模型与底层控制能够正常运行。
2. 进入 [Pupper Locomotion](./locomotion)，得到可接受速度指令的运动策略。
3. 进入 [Pupper VLA](./vla)，将感知与语言理解连接到运动技能。
4. 在 AMD 平台上记录训练、推理和完整任务的实验结果。

两个方向也可以独立学习。如果已经有可用的运动策略或技能接口，可以直接从 VLA 方向开始。

## 项目交付

完成项目后，应至少保留以下结果：

- 一份可复现的 AMD 平台实验环境与运行记录；
- 一条能够跟踪速度指令的 Pupper 运动策略；
- 一套供上层智能调用的动作或技能接口；
- 一个融合视觉、语言与动作的完整 Demo；
- 训练曲线、评估指标、演示视频与关键技术决策说明。
