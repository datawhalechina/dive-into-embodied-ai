---
sidebar_position: 0
title: "导读"
displayed_sidebar: foundationsVlaSidebar
---

# 视觉-语言-动作模型(VLA)系列导读

**先修建议**
- 具备 Transformer 与自回归建模基础。
- 熟悉视觉编码器的基本概念（如 ViT、对比学习）。
- 了解机器人控制中的状态、动作与控制频率。

**本节目标**
- 建立 VLA 系列的总体学习地图与章节关系。
- 明确本系列围绕的核心技术问题与演进主线。
- 给出按目标导向的阅读顺序，降低后续章节切入成本。

视觉-语言-动作模型（Vision-Language-Action, VLA）把图像观测、语言指令与连续控制统一到同一条策略建模链路中。本章作为导读，先说明“为什么出现 VLA、这一系列覆盖什么、应如何阅读”，为后续各章的模型与工程细节提供统一上下文。

## 1. 本系列关注什么

核心关注四个问题：

- VLA 为什么出现，它相对传统 BC/RL 解决了什么问题
- 动作表征为什么会分化为 token 化与连续生成两条路线
- 数据规模、模型结构、部署系统如何共同决定效果上限
- 从"会做"到"可控、可学、可泛化"，下一代 VLA 的关键约束是什么

## 2. 适用读者与前置知识

适合希望系统阅读 VLA 论文与工程实现的读者，尤其是需要理解 RT-1、RT-2、OpenVLA、ACT、Diffusion Policy、pi0/pi0.5/pi0.6/pi0.7 技术链路的场景。

前置建议：

- Transformer / 自回归建模基础
- 基本视觉编码器概念（ViT、对比学习）
- 机器人控制中的状态、动作、控制频率基本概念

## 3. 知识地图

```text
图像观测 + 语言指令
  -> 视觉编码与语义对齐
  -> VLM / LLM 主干
  -> 动作建模（token / diffusion / flow matching）
  -> 实时执行系统（chunking / async control）
  -> 在线数据回流与策略更新
```

## 4. 章节安排

| 章节 | 核心主题 | 主要问题 |
|------|----------|----------|
| 1 | VLA 全景与演化脉络 | 为什么 VLA 会成为具身智能主线 |
| 2 | RT-1 架构决策 | 3 Hz 与 256 bin 背后的工程权衡 |
| 3 | RT-2 语义迁移 | 为什么 VLM 语义会在机器人上涌现 |
| 4 | Open X-Embodiment | 跨机体数据如何标准化与混合 |
| 5 | OpenVLA 工程化 | 闭源范式如何被开源复现 |
| 6 | ACT 动作分块 | 如何压制长时序误差累积 |
| 7 | Diffusion Policy | 连续多模态动作分布如何建模 |
| 8 | pi0 基础模型 | VLM + 动作专家如何分工协同 |
| 9 | pi0.5 开放世界 | 跨场景泛化为何需要层级语义 |
| 10 | RTC 实时执行 | 慢推理如何嵌入硬实时控制 |
| 11 | pi0.6 在线学习 | 部署经验如何反向驱动策略提升 |
| 12 | pi0.7 可引导性 | 语言引导与组合泛化如何落地 |

## 5. 阅读建议

- 推荐顺序:`1 -> 2 -> 3 -> ... -> 12`
- 若偏工程部署:优先读 `5 -> 10 -> 11`
- 若偏动作建模:优先读 `6 -> 7 -> 8 -> 9`
- 若偏前沿趋势:优先读 `1 -> 9 -> 11 -> 12`

## 6. 章节目录

- [1. 一条从语言到动作的线](1.vla_landscape.md)
- [2. RT-1:为什么是 3 Hz,为什么是 256 个 bin](2.rt1_architecture.md)
- [3. RT-2:一个机械臂怎么学会了"把苹果放到数字 3 上"](3.rt2_semantic_transfer.md)
- [4. Open X:数据与跨机体迁移](4.openx_data_scaling.md)
- [5. OpenVLA:复现与关键取舍](5.openvla_engineering.md)
- [6. ACT:当误差以 $O(L^T)$ 增长时,你该怎么办](6.act_action_chunking.md)
- [7. Diffusion Policy:动作建模](7.diffusion_policy.md)
- [8. pi0:架构与训练配方](8.pi0_foundation_model.md)
- [9. pi0.5:开放世界与分层执行](9.pi05_open_world.md)
- [10. RTC:实时执行与系统交付](10.rtc_realtime_execution.md)
- [11. pi0.6:在线闭环与优势策略](11.pi06_online_learning.md)
- [12. pi0.7:可引导与组合泛化(终篇)](12.pi07_steerable_emergence.md)
