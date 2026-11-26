---
title: "Quadruped series"
description: "Quadruped projects: build a robot from scratch with the CS123 simulation course, explore Petoi Quaddle walking simulation, and train Quaddle VLA navigation."
sidebar_position: 1
displayed_sidebar: practicesQuadrupedSidebar
---

# Quadruped series

Explore quadruped modeling, control, reinforcement learning, and multimodal decision-making through projects covering **single-joint PD control, full-body gaits, PPO policy training, browser simulation, and language and vision feedback loops**.

## 🗂️ Project overview \{#projects}

| Category | Project | Description |
| :--- | :--- | :--- |
| Chapter-based course | [Build a quadruped from scratch](/docs/practices/quadruped/cs123/intro) | Eight chapters covering MuJoCo actuators, forward and inverse kinematics, Pupper modeling, open-loop gaits, PPO training, LLM control, and visual ball tracking |
| Cross-backend locomotion & Sim2Sim | [Unitree Go2 locomotion · MJX (ROCm / CUDA)](/docs/practices/amd/unitree-rl-mjx) | Train Go2 velocity policies with MuJoCo MJX / JAX on an 8GB AMD or NVIDIA GPU and export ONNX directly to the official C++ Sim2Sim stack (also playable in the [browser simulator](/simulation)) |
| Browser simulation and terrain experiments | [Petoi Quaddle walking simulation](/docs/practices/quadruped/quaddle-mujoco/) | Control a four-servo robot in the browser and explore closed-loop linkage adaptation, residual PPO training, and terrain experiments |
| Language and vision feedback | [Petoi Quaddle VLA navigation training](/docs/practices/vla/quaddle/) | Navigate using language instructions and camera images: SmolVLA decisions with low-level lookup-table CPG control on a 10GB GPU (L2 navigation 90%, L3 obstacle avoidance 80%, L4 composition 85%) |
| Cross-simulator validation | [Sim2Sim validation guide](/docs/practices/quadruped/sim2sim/placeholder) | Cross-engine policy transfer alignment, failure diagnostic checklists, and live Go2 / Flamingo walkthroughs |
| Hardware deployment | [Sim2Real engineering guide](/docs/practices/quadruped/sim2real-guide/placeholder) | End-to-end calibration, system identification, safety guards, and staged bring-up from simulation to quadruped hardware |

## 🎬 Demos \{#demos}

| [![CS123 quadruped gait comparison](@site/docs/practices/quadruped/cs123/figs/lab5_forward_gait_comparison.gif)](/docs/practices/quadruped/cs123/intro)<br/>**[Build a quadruped from scratch](/docs/practices/quadruped/cs123/intro)**<br/>CS123 simulation · Walk / Pace / Bound / Gallop comparison | [![PPO gait policy comparison](@site/docs/practices/quadruped/cs123/figs/lab6_rl_comparison.gif)](/docs/practices/quadruped/cs123/rl-gait)<br/>**[PPO gait policy training](/docs/practices/quadruped/cs123/rl-gait)**<br/>MuJoCo + Stable-Baselines3 · Closed-loop balance and velocity tracking |
| :---: | :---: |

### Build a quadruped from scratch · Chapter demos \{#cs123-gifs}

| [![HFE joint tracking a sine trajectory](@site/docs/practices/quadruped/cs123/figs/lab1_hfe_sine.gif)](/docs/practices/quadruped/cs123/pid-control)<br/>**[1. Actuators and PD control](/docs/practices/quadruped/cs123/pid-control)**<br/>Joint position feedback and 0.3 Hz sine tracking | [![Pupper single-leg FK joint sweep](@site/docs/practices/quadruped/cs123/figs/lab2_leg_teleop.gif)](/docs/practices/quadruped/cs123/forward-kinematics)<br/>**[2. Forward kinematics](/docs/practices/quadruped/cs123/forward-kinematics)**<br/>Three-DoF leg transforms and joint sweeps | [![Pupper single-leg Raibert stepping](@site/docs/practices/quadruped/cs123/figs/lab3_leg_stepping.gif)](/docs/practices/quadruped/cs123/inverse-kinematics)<br/>**[3. Inverse kinematics](/docs/practices/quadruped/cs123/inverse-kinematics)**<br/>Foot trajectory inversion and suspended Raibert stepping |
| :---: | :---: | :---: |
| [![Pupper dropping and holding a standing pose](@site/docs/practices/quadruped/cs123/figs/pupper-stand.gif)](/docs/practices/quadruped/cs123/quadruped-mjcf)<br/>**[4. Build a quadruped model](/docs/practices/quadruped/cs123/quadruped-mjcf)**<br/>Pupper v3 MJCF, floating base, and stable landing | [![LLM multi-step language control](@site/docs/practices/quadruped/cs123/figs/lab7_llm_sequence.gif)](/docs/practices/quadruped/cs123/llm-control)<br/>**[7. Language control](/docs/practices/quadruped/cs123/llm-control)**<br/>LLM function calling for multi-step action sequences | [![HSV detection and closed-loop ball tracking](@site/docs/practices/quadruped/cs123/figs/lab8_ball_chase.gif)](/docs/practices/quadruped/cs123/perception)<br/>**[8. Visual perception](/docs/practices/quadruped/cs123/perception)**<br/>Onboard camera detection and ball tracking with an RL gait |

### Petoi Quaddle walking simulation · Scene previews \{#quaddle-demos}

| [![Quaddle walking on an infinite checkerboard](@site/docs/practices/quadruped/quaddle-mujoco/figs/ground-long-walk.webp)](/docs/practices/quadruped/quaddle-mujoco/)<br/>**[Quaddle introduction: browser simulation and residual PPO](/docs/practices/quadruped/quaddle-mujoco/)**<br/>Run MuJoCo WebAssembly in the browser to walk, stand, and pause | [![Quaddle flat-ground stopping and turning](@site/docs/practices/quadruped/quaddle-mujoco/figs/experiments/interaction.webp)](/docs/practices/quadruped/quaddle-mujoco/experiments)<br/>**[Quaddle experiments: results and failure analysis](/docs/practices/quadruped/quaddle-mujoco/experiments)**<br/>Recorded tests and analysis of stopping, turning, slopes, steps, and uneven terrain |
| :---: | :---: |

### Petoi Quaddle VLA navigation · Previews \{#quaddle-vla-demos}

| [![L2 language navigation](@site/docs/practices/vla/quaddle/figs/quaddle_L2_nav_silky50.gif)](/docs/practices/vla/quaddle/)<br/>**[VLA navigation: train a policy on a 10GB GPU](/docs/practices/vla/quaddle/)**<br/>SmolVLA with temporal ensembling, 90% over 20 episodes of navigating to a red block | [![Original CPG gait baseline](@site/docs/practices/vla/quaddle/figs/quaddle_gait_orig_smooth.gif)](/docs/practices/vla/quaddle/experiment-notes)<br/>**[Low-level control baseline (without VLA)](/docs/practices/vla/quaddle/experiment-notes)**<br/>Open-loop lookup-table CPG gait, upright at 69.8 mm/s |
| :---: | :---: |

## 🧭 Course outline \{#outline}

### 1. Build a quadruped from scratch (CS123 simulation) \{#cs123}

A Chinese simulation course inspired by Stanford CS123, covering control, kinematics, modeling, gaits, reinforcement learning, language, and vision without requiring physical hardware. The first three chapters also include browser experiments.

| Chapter | Topics | Browser experiment |
| :--- | :--- | :--- |
| [0. Course overview](/docs/practices/quadruped/cs123/intro) | Scope, prerequisites, environment requirements, and a six-week study plan | — |
| [1. Actuators and PD control](/docs/practices/quadruped/cs123/pid-control) | Motor models, gear ratios, bang-bang control, and joint PD/PID responses | [PD experiment](/cs123/pd-playground) |
| [2. Forward kinematics](/docs/practices/quadruped/cs123/forward-kinematics) | Rigid transforms, homogeneous matrices, DH / URDF conventions, and three-DoF leg FK | [FK experiment](/cs123/fk-playground) |
| [3. Inverse kinematics](/docs/practices/quadruped/cs123/inverse-kinematics) | Geometric solutions, Jacobian iteration, and foot trajectory tracking | [IK experiment](/cs123/ik-playground) |
| [4. Build a quadruped model](/docs/practices/quadruped/cs123/quadruped-mjcf) | Pupper v3 URDF/MJCF tree, floating base, and standing parameter sweeps | — |
| [5. Gait control](/docs/practices/quadruped/cs123/gait-control) | Phase scheduling, swing and stance trajectories, and Trot / Walk / Pace / Bound / Gallop | — |
| [6. Policy training](/docs/practices/quadruped/cs123/rl-gait) | Gymnasium environments, exponential velocity rewards, and PPO gait training | — |
| [7. Language control](/docs/practices/quadruped/cs123/llm-control) | Tool APIs, function calling / ReAct loops, and voice commands | — |
| [8. Visual perception](/docs/practices/quadruped/cs123/perception) | Onboard cameras, HSV detection, pixel-to-direction mapping, and closed-loop ball tracking | — |

### 2. Petoi Quaddle walking simulation \{#quaddle}

Based on Petoi's official Quaddle model, with MuJoCo closed-loop linkage constraints, passive torsion springs, and contact parameters for real-time browser interaction.

| Chapter | Topics |
| :--- | :--- |
| [Introduction: make a quadruped walk](/docs/practices/quadruped/quaddle-mujoco/) | Real-time MuJoCo in the browser, local `mjpython`, an infinite ground plane, and a 32-dimensional residual PPO training workflow |
| [Experiments: results and failure analysis](/docs/practices/quadruped/quaddle-mujoco/experiments) | Recorded comparisons and reproduction commands for stopping and turning, 3°/5° slopes, 4 mm descents, 2 mm ascents, uneven ground, and candidate running policies |

### 3. Petoi Quaddle VLA navigation training \{#quaddle-vla}

Build a VLA pipeline in the Quaddle introduction environment: given "walk to the red block," the robot turns, distinguishes colors, approaches the target, and stops. SmolVLA 450M provides high-level decisions, with lookup-table CPG control and a 10GB GPU memory budget.

| Chapter | Topics |
| :--- | :--- |
| [VLA navigation: train a policy on a 10GB GPU](/docs/practices/vla/quaddle/) | Data collection, SmolVLA fine-tuning, and closed-loop evaluation; L2 navigation demos (90%) and capability limits |
| [Experiment notes and v5 roadmap](/docs/practices/vla/quaddle/experiment-notes) | v4/v5/v6 experiments, including negative results; temporal ensembling and 8.2x faster training |
| [Implementation status and plans](/docs/practices/vla/quaddle/progress) | L1–L4 evaluation progress, task decomposition (L3 obstacle avoidance 80%, L4 composition 85%), and hardware handoff plans |

### 4. Unitree Go2 locomotion (MJX · ROCm / CUDA) \{#go2-mjx}

A MuJoCo MJX / JAX port of Unitree's official `unitree_rl_mjlab`, running identical training code on AMD ROCm and NVIDIA CUDA GPUs and exporting ONNX policies directly to the unmodified official `unitree_mujoco` and `go2_ctrl` C++ stack.

| Chapter | Topics |
| :--- | :--- |
| [Go2 locomotion overview](/docs/practices/amd/unitree-rl-mjx) | Cross-backend baseline metrics (R9700 / A40 / 8GB VRAM), bundled policy, and repository layout |
| [1. Environment setup & hardware notes](/docs/practices/amd/unitree-rl-mjx/environment-setup) | Multi-backend `uv` installation, verified Go2 mesh fetch, AUP cloud, and RDNA4 (`gfx1201`) notes |
| [2. Train the Go2 velocity policy](/docs/practices/amd/unitree-rl-mjx/training) | 20-minute smoke run, full 980M-step training, reward plots, and offscreen GIF rendering |
| [3. ONNX export & official Sim2Sim](/docs/practices/amd/unitree-rl-mjx/sim2sim) | ONNX normalizer baking & contract verification, C++ stack build, headless FIFO testing, and tracking comparison |
