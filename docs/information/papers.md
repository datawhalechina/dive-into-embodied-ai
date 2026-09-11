---
title: 论文与研究
description: 按研究主题整理论文、项目主页、代码与中文解读。
sidebar_position: 2
displayed_sidebar: informationSidebar
---

# 论文与研究

从研究问题出发，连接原论文、作者项目主页与站内解读。首批收录与现有教程相关的工作，后续逐步补充运动控制、感知和世界模型等主题。

## 模仿学习与动作生成

### ACT · Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

**2023 · 双臂操作 / 模仿学习 / 动作分块**

研究低成本双臂硬件如何通过示教完成精细操作。ACT 一次预测一段动作，围绕长时序模仿学习中的误差累积展开设计。

- 原始来源：[论文](https://arxiv.org/abs/2304.13705) · [作者项目主页](https://tonyzhaozh.github.io/aloha/)
- 站内阅读：[ACT 架构与关键机制](/docs/foundations/vla/act_action_chunking)
- 动手实践：[ACT 双臂操作训练](/docs/practices/vla/act)

### Diffusion Policy · Visuomotor Policy Learning via Action Diffusion

**2023 · 视觉运动策略 / 扩散模型 / 连续动作**

将机器人动作生成建模为条件去噪过程，用于表达多模态动作分布，并结合视觉条件与滚动执行生成动作序列。

- 原始来源：[论文](https://arxiv.org/abs/2303.04137) · [作者项目主页](https://diffusion-policy.cs.columbia.edu/)
- 站内阅读：[Diffusion Policy 动作建模](/docs/foundations/vla/diffusion_policy)

## 视觉语言动作模型

### OpenVLA · An Open-Source Vision-Language-Action Model

**2024 · VLA / 多机器人数据 / 模型微调**

将视觉语言模型接到机器人动作预测上，公开模型与训练代码，便于研究跨任务学习和面向新机器人场景的微调。

- 原始来源：[论文](https://arxiv.org/abs/2406.09246) · [作者项目主页](https://openvla.github.io/)
- 站内阅读：[OpenVLA 复现与关键取舍](/docs/foundations/vla/openvla_engineering)
- 相关资源：[具身数据集](./datasets) · [开源项目](./open-source)

整理与来源核对：2026-09-11。
