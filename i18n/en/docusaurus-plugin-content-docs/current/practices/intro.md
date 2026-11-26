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
| [Robot arm series](/docs/practices/robot-arm/placeholder) | 11 available | Continuous control, VLA, data collection, imitation learning, ROS 2, and real arms |
| [Quadruped series](/docs/practices/quadruped/placeholder) | 6 available | Reinforcement learning, kinematics, gait control, MJX & Sim2Sim, Sim2Real, and VLA navigation |
| [Biped series](/docs/practices/humanoid/placeholder) | 9 available | Unitree G1 humanoid RL, wheeled-biped Isaac Lab + CaT, parallel GPU PPO, and curriculum learning |
| [Mobile manipulation](/docs/practices/mobile-manipulation/placeholder) | 3 available | SLAM + Nav2 navigation, vision-language navigation (VLN), and mobile base + arm coordination |

## Available projects \{#available-projects}

- [Unitree G1 humanoid locomotion (4 parts)](/docs/practices/humanoid/unitree-g1/): full walkthrough covering 16-term reward analysis and 10,000-iteration PPO dynamics, 0.2–0.8 m/s MuJoCo Sim2Sim, browser stateless LSTM ONNX deployment (live in [Simulation](/simulation)), plus AMP, Get-Up, and GR00T Whole-Body Control notes.
- [Go2 quadruped locomotion · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx): MuJoCo MJX / JAX port of Unitree's `unitree_rl_mjlab`, running identical Go2 velocity training on AMD ROCm and NVIDIA CUDA with ONNX export directly into the unmodified official C++ Sim2Sim stack.
- [Flamingo wheeled biped · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro): build an end-to-end RL locomotion pipeline for an 8-DoF wheeled biped, covering delayed/geared actuators, 4096-env PPO vs CaT, ONNX normalizer fusion, and zero-shot MuJoCo Sim2Sim stress tests (overview and 6 chapters).
- [MicroDuck RL: walking, standing up, and rolling](/docs/practices/humanoid/microduck-rl): start with MJCF, BAM actuators, and mjlab task registration; train 18 motion modes in MuJoCo Warp and inspect offscreen evaluation through GIF and MP4 recordings.

## AMD projects \{#amd}

- [AUP Learning Cloud](/docs/practices/amd/aup-learning-cloud): use Ryzen AI APUs, JupyterHub, Code Server, and ROCm in your browser for exercises, edge inference, and small experiments.
- [Go2 locomotion · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx): train Unitree Go2 velocity policies on an 8GB GPU across ROCm and CUDA, including RDNA4 hardware notes, 980M-step training, and closed-loop official Sim2Sim verification.
- [MicroDuck RL · AMD ROCm](/docs/practices/amd/microduck-rl): build ROCm Warp and MuJoCo Warp on a Radeon R9700, fix dynamic broadphase caching, and train biped PPO policies.
- [ACT bimanual training · AMD ROCm](/docs/practices/amd/vla-act): train ACT in BF16 on Radeon GPUs, resume checkpoints, evaluate 20 episodes, and export videos.
- [Explore the Pupper quadruped](/docs/practices/amd/pupper-control/intro): the AMD flagship project, with **Pupper Locomotion** for RL motion policies and **Pupper VLA** for vision-language-action intelligence.

## Simulation projects \{#simulation}

| Project | Technical path | Status |
| --- | --- | --- |
| [Build a quadruped from scratch](/docs/practices/quadruped/cs123/intro) | MuJoCo, PD control, kinematics, gaits, policy training, and perception | Overview and 8 chapters available |
| [Go2 quadruped locomotion · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx) | MuJoCo MJX / JAX, cross-backend training, ONNX export, and official Sim2Sim | Overview and 3 tutorials available |
| [Petoi Quaddle four-servo walking simulation](/docs/practices/quadruped/quaddle-mujoco) | Official URDF adaptation, browser controls, periodic gaits, and residual PPO training | Available |
| [MicroDuck RL: walking, standing up, and rolling](/docs/practices/humanoid/microduck-rl) | mjlab, MuJoCo Warp, parallel PPO on CUDA, and 18 motion modes | Available |
| [Getting started with MuJoCo](/docs/practices/robot-arm/mujoco-arm-pick-place) | MJCF, physics simulation, and Python control | Available |
| [DDPG InvertedPendulum](/docs/practices/robot-arm/ddpg-mujoco/invertedpendulum-v5) | Continuous-control basics and a DDPG baseline | Available |
| [DDPG Reacher](/docs/practices/robot-arm/ddpg-mujoco/reacher-v5) | Target tracking with a planar robot arm | Available |
| [DDPG Pusher](/docs/practices/robot-arm/ddpg-mujoco/pusher-v5) | Contact manipulation and reward design | Available |
| [ACT bimanual training](/docs/practices/vla/act) | ALOHA, imitation learning, ACT, and evaluation over multiple episodes | Available |
| [π₀.₅ + RECAP: LIBERO reproduction](/docs/practices/vla/recap) | Value models, advantage annotation, ACP, and LIBERO evaluation | Available |
| [Petoi Quaddle VLA navigation training](/docs/practices/vla/quaddle/) | SmolVLA fine-tuning, low-level CPG control, temporal ensembling, and task decomposition | Available |
| [Sim2Sim validation guide](/docs/practices/quadruped/sim2sim/placeholder) | Cross-simulator policy alignment, diagnostic checklists, and Go2 / Flamingo walkthroughs | Available |
| [Unitree G1 humanoid locomotion](/docs/practices/humanoid/unitree-g1/) | 16-term reward analysis, 10k-iter PPO curves, multi-speed MuJoCo Sim2Sim, browser LSTM, and GR00T WBC | Overview and 3 guides available |
| [Flamingo wheeled biped · Isaac Lab](/docs/practices/humanoid/flamingo-isaaclab/intro) | 8-DoF wheeled-biped balance, PPO vs CaT, ONNX export, and MuJoCo Sim2Sim benchmarking | Overview and 6 chapters available |
| [Bipedal balance and push recovery](/docs/practices/humanoid/balance-control/placeholder) | CoM / angular-momentum recovery, push testing, and live Unitree G1 browser simulation | Available |
| [Whole-body humanoid motion tracking](/docs/practices/humanoid/motion-tracking/placeholder) | Motion retargeting, reference-tracking rewards, and whole-body imitation | Available |
| [Bipedal high-level task planning](/docs/practices/humanoid/task-planning-demo/placeholder) | VLM/LLM task decomposition, skill orchestration, and locomotion safety gating | Available |
| [Mobile manipulation series (Nav / VLN / Coop)](/docs/practices/mobile-manipulation/placeholder) | SLAM + Nav2 navigation, vision-language navigation, and mobile base + arm manipulation | 3 guides available |

## Real robot projects \{#real-robots}

The LeRobot notes introduce data and toolchains; the SO-101 project then applies that knowledge to a real robot arm.

| Project | Technical path | Status |
| --- | --- | --- |
| [LeRobot course notes in Chinese](/docs/practices/robot-arm/data-collection/lerobot-course) | Robot learning, datasets, toolchains, and classical robotics | Units 0–2 compiled |
| [SO-101 + LeRobot hardware tutorial](/docs/practices/robot-arm/data-collection/so101-lerobot-real) | Hardware connectivity, safety tests, and action playback | Available |
| [ROS 2 arm control guide](/docs/practices/robot-arm/ros2-arm-control/placeholder) | `ros2_control`, MoveIt 2 planning, and safe execution | Available |
| [Custom imitation learning workflow](/docs/practices/robot-arm/imitation-learning/placeholder) | Teleoperation collection, curation, and ACT / Diffusion Policy loops | Available |
| [Sim2Real engineering deployment guide](/docs/practices/quadruped/sim2real-guide/placeholder) | System identification, domain randomization, safety guards, and staged bring-up | Available |
