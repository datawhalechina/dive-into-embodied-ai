---
title: 具身智能技术路线全景图
description: 从任务、方法与工程支撑三个视角理解具身智能，并进入交互学习地图。
sidebar_position: 2
---

# 具身智能技术路线全景图

先打开 [具身智能学习地图](/learning-map)，用具体任务查看感知、决策、控制与工程底座之间的关系。

## 用三个视角定位一项技术

| 视角 | 要回答的问题 | 示例 |
| --- | --- | --- |
| 任务 | 希望机器人完成什么？ | 抓取与操作、导航、足式运动、移动操作 |
| 方法 | 用什么方式解决问题？ | 感知与状态估计、规划与控制、模仿学习、强化学习、VLA、世界模型 |
| 工程支撑 | 怎样训练、运行并验证系统？ | 数据、仿真、评测、机器人本体、传感器、通信与 ROS2 |

这三个视角相互交叉。例如，机械臂抓取可以使用运动规划和反馈控制，也可以从示教数据中学习策略；两种方式都需要明确观测、动作与成功条件。

## 从理解走向实践

1. 在 [学习地图](/learning-map#knowledge-map) 中选择一个任务，理解完整闭环。
2. 沿 [新手路线](/learning-map#first-steps) 完成浏览器实验与最小仿真。
3. 根据 [实践方向](/learning-map#directions) 选择一条主线，记录结果并分析失败原因。
4. 遇到知识缺口时，到 [理论基础](/docs/foundations/intro) 查找对应专题。

## 参考总纲

- [Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md)：算法、基础设施、控制与硬件的领域索引。
- [PKU EPIC Lab · embodied-ai-start](https://github.com/jiangranlv/embodied-ai-start)：从任务定义、机器人技能到研究方法的入门指南。

本站据此组织学习入口与原创说明，具体内容以链接到的教程及原项目为准。
