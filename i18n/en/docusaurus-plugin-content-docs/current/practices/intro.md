---
title: Project overview
sidebar_position: 1
displayed_sidebar: practicesOverviewSidebar
---

# Projects: from simulation to real robot deployment

Choose a task and work through environment setup, training, control, and evaluation with a project, demo, or reproduction study. Follow a sequence of chapters to build a complete system, or choose a standalone experiment to test a method. For questions about principles, look up the related module in [foundations](/docs/foundations/intro). For original papers, data, and code, see the introduction's [resources](/docs/introduction/resources).

## Where to start \{#getting-started}

- To build a system from scratch: start with the [quadruped project](/docs/practices/quadruped/cs123/intro). Its eight chapters cover actuators, kinematics, modeling, gaits, and policy training. Basic Python is required; you can study control and linear algebra alongside the experiments.
- To learn about arm datasets and hardware operation: read the [LeRobot course notes in Chinese](/docs/practices/robot-arm/data-collection/lerobot-course), then move on to the [SO-101 hardware tutorial](/docs/practices/robot-arm/data-collection/so101-lerobot-real). The notes cover Units 0–2 and require basic Python; the hardware tutorial requires a compatible robot.
- To try a simulation first: choose [MicroDuck RL](/docs/practices/humanoid/microduck-rl) or [ACT bimanual manipulation](/docs/practices/vla/act), and set up the required environment.
- To master wheeled-biped balance and cross-simulator deployment: follow the [Flamingo wheeled biped · Isaac Lab course](/docs/practices/humanoid/flamingo-isaaclab/intro), whose six chapters cover parallel PPO, Constraints as Terminations (CaT), ONNX export, and MuJoCo robustness benchmarking.

Before starting, check the environment and goals. Keep your code, parameters, and results as you work, then use evaluation and failure cases to check your understanding. The project chapters are currently in Chinese; the status labels below describe the available source content.

## Directions \{#directions}

| Direction | Projects | Suitable for |
| --- | --- | --- |
| [Robot arm series](/docs/practices/robot-arm/placeholder) | 9 available, 2 planned | Continuous control, VLA, data collection, imitation learning, and real arms |
| [Quadruped series](/docs/practices/quadruped/placeholder) | 3 available, 2 planned | Reinforcement learning, kinematics, gait control, and sim-to-real |
| [Biped series](/docs/practices/humanoid/placeholder) | 5 available, 3 planned | Wheeled-biped Isaac Lab + CaT, parallel GPU PPO, and multi-stage curriculum learning |
| [Mobile manipulation](/docs/practices/mobile-manipulation/placeholder) | 3 planned | Combining navigation and manipulation |

## Available projects \{#available-projects}

- [Flamingo wheeled biped · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro): build an end-to-end RL locomotion pipeline for an 8-DoF wheeled biped, covering delayed/geared actuators, 4096-env PPO vs CaT, ONNX normalizer fusion, and zero-shot MuJoCo Sim2Sim stress tests (overview and 6 chapters).
- [MicroDuck RL: walking, standing up, and rolling](/docs/practices/humanoid/microduck-rl): start with MJCF, BAM actuators, and mjlab task registration; train 18 motion modes in MuJoCo Warp and inspect offscreen evaluation through GIF and MP4 recordings.

## AMD projects \{#amd}

- [AUP Learning Cloud](/docs/practices/amd/aup-learning-cloud): use Ryzen AI APUs, JupyterHub, Code Server, and ROCm in your browser for exercises, edge inference, and small experiments.
- [MicroDuck RL · AMD ROCm](/docs/practices/amd/microduck-rl): build ROCm Warp and MuJoCo Warp on a Radeon R9700, fix dynamic broadphase caching, and train biped PPO policies.
- [ACT bimanual training · AMD ROCm](/docs/practices/amd/vla-act): train ACT in BF16 on Radeon GPUs, resume checkpoints, evaluate 20 episodes, and export videos.
- [Explore the Pupper quadruped](/docs/practices/amd/pupper-control/intro): the AMD flagship project, with **Pupper Locomotion** for RL motion policies and **Pupper VLA** for vision-language-action intelligence.

## Simulation projects \{#simulation}

| Project | Technical path | Status |
| --- | --- | --- |
| [Build a quadruped from scratch](/docs/practices/quadruped/cs123/intro) | MuJoCo, PD control, kinematics, gaits, policy training, and perception | Overview and 8 chapters available |
| [Petoi Quaddle four-servo walking simulation](/docs/practices/quadruped/quaddle-mujoco) | Official URDF adaptation, browser controls, periodic gaits, and residual PPO training | Available |
| [MicroDuck RL: walking, standing up, and rolling](/docs/practices/humanoid/microduck-rl) | mjlab, MuJoCo Warp, parallel PPO on CUDA, and 18 motion modes | Available |
| [Getting started with MuJoCo](/docs/practices/robot-arm/mujoco-arm-pick-place) | MJCF, physics simulation, and Python control | Available |
| [DDPG InvertedPendulum](/docs/practices/robot-arm/ddpg-mujoco/invertedpendulum-v5) | Continuous-control basics and a DDPG baseline | Available |
| [DDPG Reacher](/docs/practices/robot-arm/ddpg-mujoco/reacher-v5) | Target tracking with a planar robot arm | Available |
| [DDPG Pusher](/docs/practices/robot-arm/ddpg-mujoco/pusher-v5) | Contact manipulation and reward design | Available |
| [ACT bimanual training](/docs/practices/vla/act) | ALOHA, imitation learning, ACT, and evaluation over multiple episodes | Available |
| [π₀.₅ + RECAP: LIBERO reproduction](/docs/practices/vla/recap) | Value models, advantage annotation, ACP, and LIBERO evaluation | Available |
| [Sim2Sim validation](/docs/practices/quadruped/sim2sim/placeholder) | Cross-simulator policy validation | In development |
| [Flamingo wheeled biped · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro) | 8-DoF wheeled-biped balance, PPO vs CaT, ONNX export, and MuJoCo Sim2Sim benchmarking | Overview and 6 chapters available |

## Real robot projects \{#real-robots}

The LeRobot notes introduce data and toolchains; the SO-101 project then applies that knowledge to a real robot arm.

| Project | Technical path | Status |
| --- | --- | --- |
| [LeRobot course notes in Chinese](/docs/practices/robot-arm/data-collection/lerobot-course) | Robot learning, datasets, toolchains, and classical robotics | Units 0–2 compiled |
| [SO-101 + LeRobot hardware tutorial](/docs/practices/robot-arm/data-collection/so101-lerobot-real) | Hardware connectivity, safety tests, and action playback | Available |
| [ROS2 arm control](/docs/practices/robot-arm/ros2-arm-control/placeholder) | ROS2 control and arm execution | In development |
| [Sim2Real guide](/docs/practices/quadruped/sim2real-guide/placeholder) | Deploying simulation policies and validating on hardware | In development |
