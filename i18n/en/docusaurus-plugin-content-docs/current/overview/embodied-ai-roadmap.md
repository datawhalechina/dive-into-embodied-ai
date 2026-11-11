---
title: An overview of embodied AI approaches
description: Understand embodied AI through tasks, methods, and engineering support, then find the introduction, foundations, and projects.
sidebar_position: 2
---

# An overview of embodied AI approaches

Start with the [introduction](/docs/introduction/intro) to understand robot tasks, platforms, and the relationships between perception, decision making, control, and engineering support.

## Three perspectives on a technique \{#perspectives}

| Perspective | Question | Examples |
| --- | --- | --- |
| Task | What should the robot accomplish? | Grasping and manipulation, navigation, legged locomotion, mobile manipulation |
| Method | How can we solve the problem? | Perception and state estimation, planning and control, imitation learning, reinforcement learning, VLA, world models |
| Engineering support | How do we train, run, and validate the system? | Data, simulation, evaluation, robot bodies, sensors, communication, ROS2 |

These perspectives overlap. For example, a robot arm can grasp objects using motion planning and feedback control, or by learning a policy from demonstrations. Both approaches need clearly defined observations, actions, and success criteria.

## From understanding to practice \{#practice}

1. Start with the [introduction](/docs/introduction/intro) to understand the complete loop of perception, decision making, and action.
2. Choose a project organized into chapters or a standalone experiment in [projects](/docs/practices/intro). Record the results and analyze failures.
3. When you find a gap in your knowledge, look up the relevant topic in [foundations](/docs/foundations/intro).
4. Open [Simulation](/simulation) to control a robot with keyboard or on-screen buttons, or switch to a showcase of trained Go2/Pupper v3 policies and the Quaddle periodic gait. MicroDuck supports manual head and joint controls.

For ROS2, start with the [tutorial overview](/docs/foundations/robotics-and-ros2/intro). The [companion code](https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/codes/foundations/robotics-and-ros2/README.en.md) includes standalone kinematics scripts and ROS reference packages; generate the MoveIt configuration in lesson 10.

## Reference guides \{#references}

- [Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide/blob/main/README.md): an index of algorithms, infrastructure, control, and hardware.
- [PKU EPIC Lab · embodied-ai-start](https://github.com/jiangranlv/embodied-ai-start): an introductory guide from task definitions and robot skills to research methods.

This site uses these references to organize learning entry points and original explanations. Consult the linked tutorials and original projects for details.
