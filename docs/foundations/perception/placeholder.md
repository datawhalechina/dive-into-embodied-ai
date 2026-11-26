---
title: 定位、触觉与感知系统总览
sidebar_position: 1
displayed_sidebar: foundationsOverviewSidebar
description: "具身智能感知系统全景：本体感知、视觉/深度/LiDAR 定位、触觉与足端接触估计、多传感器融合及标定学习路线。"
---

# 定位、触觉与感知系统总览

机器人不是直接生活在干净的仿真状态向量里，而是生活在一组**带时间戳、带坐标系、带噪声与延迟的传感器数据流**中。感知系统既负责理解外部三维世界（视觉、深度、点云、语义），也负责精确估计机器人自身的状态与接触力（本体感知、状态估计、触觉）。

## 1. 具身感知的三层架构 \{#architecture}

| 感知层级 | 典型传感器 | 输出信息 | 核心作用 |
| :--- | :--- | :--- | :--- |
| **本体感知（Proprioception）** | 关节磁编码器、IMU（陀螺仪+加速度计）、电机相电流 | 关节角/角速度、机身角速度、重力投影、估计力矩 | 50–1000 Hz 运动控制与状态估计的绝对底座 |
| **接触与触觉（Contact & Tactile）** | 足端压力开关、六维力/力矩传感器、视触觉（GelSight 等） | 足端触地状态、接触法向/切向力、滑移与局部几何 | 腿足步态切换、防打滑、机械臂柔顺装配与防捏碎 |
| **外感知（Exteroception）** | RGB / RGB-D 相机、3D LiDAR、事件相机 | 图像、深度图、局部高程图、三维点云与语义特征 | 越障感知、SLAM 定位建图、目标检测与 VLA 决策输入 |

## 2. 定位与状态估计核心问题 \{#localization-state-estimation}

1. **腿足与移动底盘状态估计**：高速运动时纯轮式/足端里程计会因打滑累积漂移，纯 IMU 积分几秒即发散。工程上通常通过 **EKF / UKF 或因子图（Factor Graph）** 融合高频 IMU（200–1000 Hz）、足端运动学接触约束与低频视觉/LiDAR 里程计（10–30 Hz）。
2. **特权信息与隐式估计（RL 路线）**：在强化学习运控中，策略往往无法直接获得精确的基座线速度与地面摩擦系数；常用 **非对称 Actor-Critic** 或 **历史观测编码（RMA / Teacher-Student）** 让策略从最近若干帧本体观测中隐式推断速度与地形特性。
3. **时间戳与坐标系对齐**：哪怕 20–50 ms 的时钟偏移或几度的相机-IMU 外参误差，就会让动态抓取偏离目标、让高程图与足端落点错位——这是 Sim2Real 中最隐蔽的掉点根源之一。

## 3. 触觉感知与接触丰富操作 \{#tactile}

在插拔、拧盖、抓取易碎物或盲走复杂地面时，纯视觉往往被手部或车身遮挡：

- **足端接触估计**：无力传感器的四足常通过**广义动量观测器**或电机电流残差估计接触力，结合足端加速度突变判断触地瞬间。
- **操作触觉**：
  - *低维阵列 / 力矩传感器*：直接输出 6 维力/力矩（$F_x, F_y, F_z, \tau_x, \tau_y, \tau_z$），与 [阻抗与力控](/docs/foundations/controllers/impedance-force-control) 天然衔接；
  - *视触觉传感器（如 GelSight / DIGIT）*：用微型相机拍摄弹性体表面形变，输出高分辨率接触纹理与滑移向量，可作为多模态编码器直接输入模仿学习或 VLA 策略。

## 4. 站内感知专题学习路线 \{#learning-path}

本板块已上线 4 篇工程深度专题 + 1 篇多模态视觉专题，建议按问题驱动顺序阅读：

| 专题 | 解决的核心问题 | 交互/延伸 |
| :--- | :--- | :--- |
| [1. 传感器标定与 sim2real](./sensor-calibration-sim2real) | 坐标系变换、时间同步与外参误差如何破坏真机闭环 | [传感器标定交互实验](/perception/sensor-calibration-playground) |
| [2. ODO、时间戳与状态估计](./odo-timestamps-state-estimation) | 里程计漂移、异频传感器对齐与融合状态估计工程实现 | — |
| [3. 相机、LiDAR 与 IMU 外参与投影](./sensor-projection) | 针孔模型、畸变校正、点云投影到图像与外参手眼标定 | — |
| [4. 在线标定、评测与故障诊断](./online-calibration-diagnosis) | 运行期漂移检测、在线外参微调与常见感知故障定位清单 | — |
| [视觉语言大模型（VLM）](/docs/foundations/vlm/intro) | ViT 视觉表征、图文对齐与多模态理解如何衔接到 VLA | [VLA 大模型专题](/docs/foundations/vla/vla-intro) |

## 关联实战 \{#related-practices}

- [CS123 第 8 章：视觉感知与闭环追球](/docs/practices/quadruped/cs123/perception)
- [Petoi Quaddle VLA 导航训练](/docs/practices/vla/quaddle/)
- [移动操作与导航路线图](/docs/practices/mobile-manipulation/placeholder)
