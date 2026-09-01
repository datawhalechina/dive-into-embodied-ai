---
title: AMD 专区
sidebar_position: 1
displayed_sidebar: practicesAmdSidebar
---

# AMD 专区

探索具身智能算法在 AMD 平台上的训练、推理与机器人应用。专区围绕可复现的完整项目组织内容，覆盖环境配置、算法实验、结果验证与部署流程。

## 云端算力

### ☁️ [AUP Learning Cloud 云算力](./aup-learning-cloud)

通过浏览器使用基于 **AMD Ryzen AI APU** 的 JupyterHub、Code Server 与 ROCm 开发环境，无需在本地配置 AMD 硬件即可完成课程练习、环境验证和小规模具身智能实验。

`AMD`　`Ryzen AI`　`ROCm`　`JupyterHub`　`Code Server`

## 旗舰项目

### 🐥 [MicroDuck RL｜AMD ROCm](./microduck-rl)

在 Radeon AI PRO R9700 上从源码编译 ROCm Warp / MuJoCo Warp，加入动态接触、graph capture 与 API 兼容回归，再运行 MicroDuck 双足 PPO smoke train 和固定 seed 长训。

`AMD`　`ROCm`　`MuJoCo Warp`　`mjlab`　`PPO`　`Humanoid`

### 🦾 [ACT 双臂操作训练｜AMD ROCm](./vla-act)

在 Radeon AI PRO R9700 上用 BF16 训练 ACT，完成 ALOHA Transfer Cube 双臂方块
交接。教程包含 ROCm 环境、10k 快速训练、断点续训、20 回合闭环评测与 MP4/GIF。

`AMD`　`ROCm`　`LeRobot`　`ACT`　`ALOHA`

### 🐕 [玩转 Pupper 四足机器人](./pupper-control/intro)

在 AMD 平台上探索 Pupper 的 **强化学习运动策略**与 **VLA 具身智能**，通过两个完整实验体验从自主运动到视觉语言动作控制的实现过程。

`AMD`　`Pupper`　`RL Locomotion`　`VLA`

- [Pupper Locomotion｜强化学习运动策略](./pupper-control/locomotion)：训练能够稳定跟踪速度指令的 locomotion policy。
- [Pupper VLA｜视觉-语言-动作智能](./pupper-control/vla)：融合视觉感知、语言理解与运动技能，完成端到端任务 Demo。

基础的 PD 控制、正逆运动学与 Pupper 搭建不在 AMD 专区重复展开，统一参考 [《从零到一搭建四足机器人》](/docs/practices/quadruped/cs123/intro)。
