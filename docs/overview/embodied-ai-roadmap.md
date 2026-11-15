---
title: 具身智能技术路线全景图
description: 从任务、方法与工程支撑三个视角理解具身智能，并找到导论、理论基础与项目实战入口。
sidebar_position: 2
---

# 具身智能技术路线全景图

先阅读 [具身导论](/docs/introduction/intro)，了解机器人任务、载体，以及感知、决策、控制与工程底座之间的关系。

## 用三个视角定位一项技术 \{#perspectives}

| 视角 | 要回答的问题 | 示例 |
| --- | --- | --- |
| 任务 | 希望机器人完成什么？ | 抓取与操作、导航、足式运动、移动操作 |
| 方法 | 用什么方式解决问题？ | 感知与状态估计、规划与控制、模仿学习、强化学习、VLA、世界模型 |
| 工程支撑 | 怎样训练、运行并验证系统？ | 数据、仿真、评测、机器人本体、传感器、通信与 ROS2 |

这三个视角相互交叉。例如，机械臂抓取可以使用运动规划和反馈控制，也可以从示教数据中学习策略；两种方式都需要明确观测、动作与成功条件。

## 从理解走向实践 \{#practice}

1. 从 [具身导论](/docs/introduction/intro) 开始，理解感知、决策与行动的完整闭环。
2. 到 [项目实战](/docs/practices/intro) 选择分章项目或独立实验，记录结果并分析失败原因。
3. 遇到知识缺口时，到 [理论基础](/docs/foundations/intro) 查找对应专题；强化学习中的符号与概念可查阅 [术语与公式速查](/docs/foundations/rl-for-robotics/quick-reference)，MDP 和时序差分章还提供就地展开的通俗解释与手算例子。
4. 在 [在线仿真](/simulation) 用键盘或屏幕按钮操纵机器人，或切换到策略展示，观察 Go2、Pupper v3 的训练策略与 Quaddle 周期步态；小黄鸭支持头部和关节自由操纵。

学习 ROS2 时，从[教程总览](/docs/foundations/robotics-and-ros2/intro)进入；[配套代码](https://github.com/datawhalechina/dive-into-embodied-ai/tree/master/codes/foundations/robotics-and-ros2)包含独立运动学脚本和 ROS 参考包，MoveIt 配置需按第 10 讲生成。

## 参考总纲 \{#references}

- [Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md)：算法、基础设施、控制与硬件的领域索引。
- [PKU EPIC Lab · embodied-ai-start](https://github.com/jiangranlv/embodied-ai-start)：从任务定义、机器人技能到研究方法的入门指南。

本站据此组织学习入口与原创说明，具体内容以链接到的教程及原项目为准。
