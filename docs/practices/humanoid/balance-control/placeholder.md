---
title: "双足动态平衡与抗扰控制"
sidebar_position: 2
displayed_sidebar: practicesHumanoidSidebar
description: "双足与轮腿机器人动态平衡原理与实战：倒立摆与捕获点（Capture Point）、外力推撞恢复训练、定量抗扰压测与浏览器实时推撞实验。"
---

# 双足动态平衡与抗扰控制

与四足拥有宽大的四点支撑多边形不同，双足和两轮足机器人本质上是**欠驱动倒立摆**：支撑面积窄、质心高、外力推撞或轻微地面起伏都会迅速放大倾倒力矩。本章从经典平衡直觉出发，串联强化学习抗扰动训练与定量压测方法。

## 1. 从倒立摆到捕获点（Capture Point） \{#lipm-capture-point}

理解双足平衡最直观的模型是**线性倒立摆模型（LIPM）**：将整机简化为高度 $z_c$ 处的质心，忽略角动量变化时，水平加速度满足：

$$\ddot{x} = \omega_0^2 (x - p_x), \quad \omega_0 = \sqrt{\frac{g}{z_c}}$$

其中 $p_x$ 为足底压力中心（CoP / ZMP）。当质心受外力推撞获得速度 $\dot{x}$ 时，为了让机器人停稳而不摔倒，下一步必须把支撑脚落到**瞬时捕获点（Instantaneous Capture Point, ICP / DCM）**：

$$\xi = x + \frac{\dot{x}}{\omega_0}$$

这一公式直接对应双足抗扰动的三种策略层级：

| 平衡策略 | 适用扰动强度 | 物理机制 | 局限 |
| :--- | :---: | :--- | :--- |
| **踝策略（Ankle Strategy）** | 小扰动 | 调整踝关节力矩移动足底压力中心 $p_x$ | 受限于脚掌长度（通常仅数厘米） |
| **髋/质心角动量策略（Hip / Momentum）** | 中等扰动 | 摆动躯干、髋部或手臂产生反作用角动量 | 受限于关节行程与角速度上限 |
| **迈步恢复策略（Stepping / Capture）** | 大扰动 | 主动跨步将支撑点移至捕获点 $\xi$ 附近 | 需要快速抬脚与精准落点控制 |

## 2. 强化学习抗扰训练的核心配方 \{#rl-push-recovery}

现代双足与轮腿策略通常通过**显式外力推撞 + 约束/姿态塑形 + 域随机化**自发涌现出“踝—髋—迈步”三级联动恢复行为：

1. **周期性速度冲量推撞（Velocity Push）**：
   - 训练中每 `3–6 s` 对基座随机施加瞬时水平速度扰动（如 MicroDuck 采用 `±0.3 m/s`，大型人形常用 `±0.5–1.0 m/s`）。
   - **关键经验**（来自 [MicroDuck 实战调参记录](/docs/practices/humanoid/microduck-rl/experiment-notes)）：单次推撞幅度不宜远高于机器人最大步行速度，否则策略会收敛成过度紧张、大步幅的“惊弓之鸟”步态；应在稳态步态建立后配合课程逐步放宽推撞强度。
2. **姿态直立与角速度惩罚**：
   - 通过 `projected_gravity` 惩罚机身前后/左右倾角，并叠加 `base_ang_vel` 阻尼项抑制高频晃动。
   - 对欠驱动轮足（如 Flamingo），结合 **约束式强化学习（CaT）** 在触碰安全倾角或关节速度边界时按概率软终止，可显著减少奖励权重调参负担（详见 [Flamingo 第 3 章：约束式强化学习 CaT](/docs/practices/humanoid/flamingo-isaaclab/cat-constrained-rl)）。
3. **质心与摩擦随机化**：
   - 质心偏移（CoM DR）强迫策略学会在偏心负载下保持平衡；地面摩擦随机化防止策略依赖“无限抓地力”硬扛推力，而是学会在打滑前主动跨步卸力。

## 3. 定量抗扰动压测协议 \{#benchmarking}

评估一个平衡策略不能只看训练回报，建议固定以下四项定量压测指标（完整代码范例见 [Flamingo 第 5 章：鲁棒性与抗扰动评测](/docs/practices/humanoid/flamingo-isaaclab/robustness-eval)）：

| 压测维度 | 测试方式 | 核心观测指标 |
| :--- | :--- | :--- |
| **冲量生存边界** | 从前/后/左/右四个方向分级递增瞬时速度或力脉冲（如 `0.2 → 1.2 m/s`） | 各方向最大不摔倒冲量、恢复时间（回到稳态误差 `< 5°` 耗时） |
| **姿态峰值与恢复** | 记录推撞后 2 秒内机身 `roll / pitch` 与竖直高度 `z` 轨迹 | 峰值倾角、最小机身高度、是否触发自碰撞 |
| **执行器余量** | 统计推撞恢复瞬间的关节峰值力矩与转速 | 力矩饱和比例（饱和过高在真机上易过流保护） |
| **零速扰动后稳态** | 静止站立被推后，指令保持 `[0, 0, 0]` | 跨步恢复后能否重新停稳而不持续漂移 |

## 4. 立刻动手实验 \{#hands-on}

- **零配置浏览器体验**：打开 [在线仿真实验室](/simulation)，选择 **Unitree G1（12-DoF 人形）**、**Unitree Go2** 或 **Pupper v3**，点击工具栏 **外力推撞**（或按 `P`）施加瞬时水平速度冲量，直接观察双足与四足 RL 策略的跨步与姿态恢复。
- **轮足完整实战**：按顺序完成 [Flamingo 6 章实战](/docs/practices/humanoid/flamingo-isaaclab/intro)，重点复现第 3 章（CaT 约束）与第 5 章（MuJoCo 零样本抗扰压测）。
- **双足多动作与起身**：进入 [MicroDuck RL](/docs/practices/humanoid/microduck-rl/)，对比行走、摔倒起身（Standup）与 12 步跳墙课程中的动态恢复策略。
- **理论底座**：[控制器：PID → LQR → MPC](/docs/foundations/controllers/intro) · [PPO 强化学习控制](/docs/foundations/rl-for-robotics/ppo)
