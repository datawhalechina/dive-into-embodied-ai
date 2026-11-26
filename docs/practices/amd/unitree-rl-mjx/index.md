---
title: Go2 四足运控｜MJX（ROCm / CUDA）
sidebar_position: 1
displayed_sidebar: practicesAmdSidebar
description: "基于 MuJoCo MJX / JAX 的 Unitree Go2 强化学习运动控制：同一份代码支持 AMD ROCm 与 NVIDIA CUDA 训练，导出 ONNX 直接接入官方 C++ 仿真部署栈。"
---

# Go2 四足运控｜MJX（ROCm / CUDA）

本项目将宇树官方 RL 训练套件 `unitree_rl_mjlab`（原依赖 CUDA-only 的 MuJoCo Warp）移植到 **MuJoCo MJX / JAX**，使**同一份训练代码和依赖锁文件**可以在 **AMD（ROCm）** 与 **NVIDIA（CUDA）** GPU 上完成收敛一致的训练，并将导出的 ONNX 策略直接装入**未改动的官方 C++ 部署 / Sim2Sim 栈**。

`Unitree Go2`　`MJX`　`JAX`　`AMD ROCm`　`CUDA`　`Sim2Sim`

:::tip[先零配置试玩训练好的 Go2 策略]
不想先配环境？可以直接前往 [在线仿真实验室](/simulation) 切换到 **Unitree Go2**，在浏览器中用键盘或触屏实时体验本仓库导出的 50 Hz PPO 行走策略。
:::

## 项目亮点与实测基线 \{#highlights}

源码位于仓库 `codes/practices/amd/unitree-rl-mjx/`，逐奖励项对齐上游参考提交 `unitree_rl_mjlab@1425b15f`，当前覆盖 **Go2 · 平地速度跟踪**任务：

| 维度 | 实测结果 |
| :--- | :--- |
| **硬件与后端** | AMD Radeon AI PRO R9700（ROCm 7.2.4）/ NVIDIA A40 & RTX 3090（CUDA 12）/ CPU 测试 |
| **显存需求** | 4096 并行环境峰值显存：ROCm 约 **4.6 GB**，CUDA 约 **1.5 GB**（**8 GB 显存即可训练**） |
| **训练耗时** | 冒烟训练（4000 万步）约 20 分钟；完整 9.8 亿步：R9700 约 **7.9 小时**，A40 约 **5.5 小时**（留出指标统计不可区分） |
| **部署闭环** | 归一化烘焙进 ONNX 图 + `deploy.yaml` 契约校验（JAX 与 ONNX 逐元素 `|Δ| < 1e-5`），直通官方 `unitree_mujoco` + `go2_ctrl` |
| **成品策略** | 仓库附带预训练成品 `policies/go2/velocity/v0/`（ONNX + `deploy.yaml`），可直接跳过训练跑 Sim2Sim |

## 教程路线 \{#roadmap}

按下面 3 步顺序即可完整走通环境搭建、GPU 并行训练与跨仿真闭环：

| 章节 | 核心内容 |
| :--- | :--- |
| [1. 环境安装与硬件约束](./environment-setup.md) | 用 `uv` 安装 ROCm / CUDA / CPU 后端、SHA-256 校验拉取 Go2 可视网格、AUP 云端运行与 RDNA4（`gfx1201`）已知约束处理 |
| [2. 训练 Go2 速度策略](./training.md) | 20 分钟冒烟验证、9.8 亿步完整训练、输出文件说明、训练曲线绘制与离线渲染 GIF |
| [3. ONNX 导出与官方栈 Sim2Sim](./sim2sim.md) | 导出 ONNX 并执行数值合同校验、编译未修改的官方 `unitree_mujoco` 与 `go2_ctrl` C++ 栈、无手柄 FIFO 脚本化测试与跟踪误差实测 |

## 仓库目录速览 \{#repo-layout}

```text
codes/practices/amd/unitree-rl-mjx/
├── src/unitree_rl_mjx/   # 环境、任务配置、训练入口与策略导出
├── tests/                # CPU 可运行的 pytest 套件
├── benchmarks/           # 吞吐 / 收敛 / Sim2Sim 度量与已发布结果（含 REPRODUCE.md）
├── policies/go2/velocity/# 预训练策略（ONNX + deploy.yaml），可直接装进官方部署栈
├── notebooks/            # 端到端一键演示 notebook（go2_training_demo.ipynb）
└── scripts/              # fetch_go2_meshes.sh、setup_rocm.sh / setup_cuda.sh / setup_cpu.sh
```

> 延伸阅读：想进一步了解跨物理引擎迁移的通用排查方法，可结合查看 [四足 Sim2Sim 验证路线图](/docs/practices/quadruped/sim2sim/placeholder) 与 [Flamingo ONNX 导出与 MuJoCo Sim2Sim](/docs/practices/humanoid/flamingo-isaaclab/onnx-and-sim2sim)。
