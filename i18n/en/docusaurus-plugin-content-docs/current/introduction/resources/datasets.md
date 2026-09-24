---
title: Embodied AI datasets
description: Robot demonstration data and evaluation benchmarks, with data types, uses, and official download pages.
sidebar_position: 10
displayed_sidebar: introductionSidebar
---

# Embodied AI datasets

This page collects data resources and evaluation benchmarks for robot learning. Before choosing one, check the robot form, the observation and action definitions, and the task scope, then decide whether it fits your training or evaluation goals.

## Real-robot data \{#real-robot-data}

### Open X-Embodiment (OXE) \{#open-x-embodiment}

Pools manipulation trajectories from many robot platforms in a standardized data format. Useful for understanding cross-robot data mixing and generalist policy training.

- Data type: real-robot manipulation trajectories; observation, action, and language fields vary by sub-dataset.
- Where to get it: [official project page and data catalog](https://robotics-transformer-x.github.io/). Find each sub-dataset through the Data link on the project page.
- Check before use: sub-dataset documentation, action spaces and sampling rates, download sizes, and each dataset's terms of use.
- Read on this site: [Open X-Embodiment and data scaling](/docs/foundations/vla/openx_data_scaling).

## Simulation data and benchmarks \{#simulation-benchmarks}

### LIBERO \{#libero}

Organizes manipulation tasks around lifelong robot learning and knowledge transfer, and provides simulation environments, task suites, and demonstrations for training and comparing robot manipulation policies.

- Data type: simulated manipulation tasks and demonstration trajectories.
- Where to get it: [official code, data downloads, and task descriptions](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Check before use: task suites, training and test splits, environment versions, and the policies and observation settings used for evaluation.

## What to check in dataset documentation \{#reading-dataset-docs}

| Aspect | What to confirm |
| --- | --- |
| Robot and task | Single arm, two arms, or another robot; grasping, manipulation, locomotion, or another task |
| Observations | Camera views, images, joint states, language instructions, and how they are aligned in time |
| Actions | Joint positions or end-effector poses, absolute or delta control, units, and frequency |
| Training and evaluation | Data splits, repeated trajectories, evaluation protocols, and task success conditions |
| Access and use | Format, size, download method, license, and citation requirements |

To learn how to read data and use the toolchain, see the [LeRobot course notes in Chinese](/docs/practices/robot-arm/data-collection/lerobot-course).

Compiled and sources checked: 2026-09-11.
