---
title: 理论技能树概述
sidebar_position: 1
displayed_sidebar: foundationsOverviewSidebar
---

# 理论技能树：按系统能力补齐具身智能工程底座

理论技能树围绕后续项目实践需要什么就补什么，让学习者按“大脑、小脑、感知系统、工程底座”的方式定位能力缺口，再进入已有专题连续学习。

还不了解这个领域的全貌，可以先读 [具身导论](/docs/introduction/intro)：它介绍具身智能的概念、发展脉络、机器人的技能、具身智能的载体、核心挑战、技术栈与就业岗位。

## 当前可学习模块

| 技能树位置 | 当前模块 | 核心问题 | 服务于 |
|------|------|---------|--------|
| 大脑：智能决策 | [强化学习与控制](./rl-for-robotics/intro) | RL 怎么用在机器人上?含模仿学习(数据驱动的策略学习) | 四足、双足、机械臂项目 |
| 大脑：智能决策 | [视觉-语言-动作大模型(VLA)](./vla/vla-intro) | 当前具身大模型在做什么? | VLA 相关项目 |
| 大脑：智能决策 | [世界模型(World Model)](./world-model/intro) | 世界模型在具身场景下怎么落地? | VLA 相关项目 |
| 小脑：运动控制 | [强化学习控制](./rl-for-robotics/ppo) | 策略学习如何接到连续控制和机器人任务上? | 四足、双足、机械臂项目 |
| 小脑：运动控制 | [控制器](./controllers/intro) | 如何从 PID、LQR 学到 MPC、阻抗控制和 ROS2 / 仿真集成? | 关节控制、轨迹跟踪、移动机器人项目 |
| 小脑：运动控制 | [运动规划](./robotics-and-ros2/moveit2_basics) | 如何从模型、坐标树进入 MoveIt 2 规划闭环? | 机械臂、移动操作项目 |
| 感官：感知系统 | [视觉语言大模型(VLM)](./vlm/intro) | 多模态模型如何理解图像与语言? | VLA 相关项目 |
| 感官：感知系统 | [定位、触觉与传感器标定](./perception/placeholder) | 机器人如何理解外部环境、自身接触状态和传感器可信度? | 移动操作、四足、人形项目 |
| 感官 / 工程底座 | [传感器标定与 sim2real](./perception/sensor-calibration-sim2real) | 坐标系、时间戳和外参误差如何影响真机闭环? | 真机部署、sim2real、数据回放 |
| 工程底座 | [仿真工具](./simulation/intro) | 怎么在仿真中跑机器人？ | 所有实践项目 |
| 工程底座 | [ROS2](./robotics-and-ros2/intro) | 坐标变换、FK/IK、tf2、URDF 与 MoveIt 2 如何串成系统? | 机械臂、四足、所有实践项目 |
| 工程底座 | [CAN 与 MCU 通信](./communication/can-mcu) | 上位机、下位机和执行器如何稳定通信? | 真机部署、硬件适配 |
| 工程底座 | [机械结构](./hardware/placeholder) | 连杆、关节、电机和末端执行器如何组成机器人本体? | 本体选型、真机项目 |
| 工程底座 | [模仿学习](./rl-for-robotics/imitation-learning) | 如何从示教数据训练策略? | LeRobot、VLA、机械臂项目 |
