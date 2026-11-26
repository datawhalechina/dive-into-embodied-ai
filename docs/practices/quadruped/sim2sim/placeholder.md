---
title: "Sim2Sim 验证实战指南"
sidebar_position: 3
displayed_sidebar: practicesQuadrupedSidebar
description: "腿足机器人跨物理引擎策略迁移（Isaac Lab / MJX / Isaac Gym → MuJoCo / 官方部署栈 / 浏览器 WASM）的系统验证方法、6 大掉点排查清单与本仓库实测对照。"
---

# Sim2Sim 验证实战指南

在把强化学习策略直接烧录到真机之前，**Sim2Sim（跨仿真器零样本迁移）** 是成本最低、暴露工程隐性 Bug 最有效的“压力测试关”。如果在两个物理引擎之间策略就已经跑偏、抖动甚至摔倒，直接上真机几乎必然失败。

## 1. 为什么要做 Sim2Sim？ \{#why-sim2sim}

训练仿真器（如 Isaac Lab、MuJoCo MJX、MuJoCo Warp）为了吞吐最大化，常使用简化的碰撞体、较少的求解器迭代次数和理想化的同步控制步；而部署仿真器或真机运行时（如原生 MuJoCo C++、`unitree_mujoco` + DDS、或本项目的浏览器 MuJoCo WASM）在**接触求解、执行器路径、传感器顺序和通信延迟**上更接近真机。

本仓库目前已经打通了多条可复现的真实 Sim2Sim 链路：

| 迁移链路 | 载体 | 策略格式 | 仓库实战入口 |
| :--- | :--- | :--- | :--- |
| **MuJoCo MJX / JAX → 官方 `unitree_mujoco` + C++ 控制器** | Unitree Go2 | ONNX（归一化烘焙）+ `deploy.yaml` | [Go2 ONNX 导出与官方栈 Sim2Sim](/docs/practices/amd/unitree-rl-mjx/sim2sim) |
| **Isaac Lab（PhysX 4096 环境）→ 原生 MuJoCo** | Flamingo 两轮足 | ONNX（归一化融合） | [Flamingo 策略导出与 MuJoCo Sim2Sim](/docs/practices/humanoid/flamingo-isaaclab/onnx-and-sim2sim) |
| **SB3 PyTorch（Python MuJoCo）→ 浏览器 MuJoCo WASM** | Pupper v3 | ONNX + `config.json` + 256 组参考对齐 | [在线仿真模型说明](/simulation) & `scripts/export-pupper-web.py` |

## 2. 六大高频 Sim2Sim 掉点根源与排查清单 \{#pitfalls}

实测中 90% 的 Sim2Sim 失败并非策略泛化差，而是**契约对齐错误**。按下面顺序逐项排查：

### ① 关节与执行器顺序错位（头号杀手）
- 训练框架（Isaac Lab / MJX）的关节顺序往往按 BFS 树或自定义列表排列（例如 Go2 训练用 `FL / FR / RL / RR`），而官方 SDK 或 MuJoCo XML 的 `<actuator>` 顺序可能完全不同（如 `FR / FL / RR / RL`）。
- **验证标准**：绝不靠数组下标硬编码，一律按**关节名查 ID**建立 `policy_idx ↔ sim_joint_id ↔ actuator_id` 显式映射表。

### ② 观测归一化、缩放与坐标系符号
- 检查以下高频差异：
  - **四元数顺序**：MuJoCo 为 `[w, x, y, z]`，而部分框架或 ROS 为 `[x, y, z, w]`；
  - **角速度坐标系**：MuJoCo 自由关节 `qvel[3:6]` 为**机身局部坐标系**，与世界系角速度不同；
  - **重力投影向量**：确认旋转矩阵转置方向（静止直立时应为 `[0, 0, -1]`）；
  - **归一化统计量**：训练时若开启了 `EmpiricalNormalization` / `VecNormalize`，务必将均值和方差烘焙进 ONNX 计算图（`Sub` / `Div` 节点），避免部署端漏做或重复归一化。

### ③ 数值合同验证（Numerical Contract）
在跑任何物理闭环前，先用固定观测批量对比“训练框架推理输出”与“部署运行时 ONNX 输出”：

```python
# 导出阶段强门禁：任意一组代表性观测上，最大逐元素绝对误差必须 < 1e-5
max_diff = np.max(np.abs(torch_or_jax_action - onnx_action))
assert max_diff < 1e-5, f"Export contract mismatch: max_diff={max_diff}"
```

本仓库的 `unitree_rl_mjx.export.install` 与 `scripts/export-pupper-web.py` 均强制执行此门禁；若数值差异超过 `1e-5`，先排查算子精度、输入维序或遗漏的激活/裁剪层。

### ④ 控制周期、Decimation 与底层 PD 路径
- 训练步长（如 `dt = 0.002 s`，`decimation = 10` → 50 Hz 策略）必须与部署仿真严格匹配。
- 区分**位置伺服（`<position>`）**与**显式力矩环（`<motor>` + 外部 PD）**：若训练使用 `tau = kp * (q_des - q) - kd * dq` 并按 `ctrlrange` 裁剪，部署端必须使用完全相同的 `kp`、`kd`、`default_angles` 与 `action_scale`。

### ⑤ 物理模型与接触求解差异
- **真实案例（Go2 MJX → 官方 `unitree_mujoco` 实测）**：同一 checkpoint 去除前 2 秒瞬态后，前向 `vx = 1.0 m/s` 从 MJX 内的 `0.93` 迁移至官方栈的 `0.85`（迁移良好），但横移 `vy = 0.4` 从 `0.30` 降至 `0.18`、原地转 `wz = 0.8` 从 `0.81` 降至 `0.41`。
- 根因包括：官方场景采用全碰撞网格、求解器迭代参数不同、以及底层 1 kHz PD 与训练环境航向伺服语义存在差异（详见 [Go2 Sim2Sim 实测分析](/docs/practices/amd/unitree-rl-mjx/sim2sim#sim2sim-results)）。

### ⑥ 初始状态与重置逻辑
- 重置时不仅要重置 `qpos` / `qvel` 到标称站姿高度和默认关节角，还必须清零**历史动作缓冲区（`last_action`）**、**步态相位计数器**以及 **LSTM 隐状态（`h, c`）**，并先调用一次 `mj_forward` 再构造首帧观测。

## 3. 自动化 Sim2Sim 验收标准 \{#acceptance}

一份合格的 Sim2Sim 验收不应只看一段视频，建议固化成自动化脚本（可参考本仓库 `scripts/check-simulation.mjs` 与 `scripts/check-simulation-models.mjs`），至少覆盖 5 组指令：

| 测试工况 | 验收项 | 典型通过阈值（四足参考） |
| :--- | :--- | :--- |
| **直行前进**（如 `vx = +0.6`） | 6–10 s 内稳定推进、无摔倒、无 `NaN` | `x > 1.0 m`，机身竖直分量 `R_zz > 0.80` |
| **倒退**（如 `vx = -0.3`） | 负向稳定位移、姿态不后仰倾覆 | `x < -0.4 m`，`R_zz > 0.80` |
| **侧向横移**（如 `vy = +0.3`） | 横向有效移动、不原地打转 | `y > 0.3 m`，`R_zz > 0.80` |
| **原地转向**（如 `wz = +0.6`） | 偏航角单调变化、机身中心漂移可控 | 偏航明显变化，`R_zz > 0.80` |
| **零速站立**（`[0, 0, 0]`） | 站姿稳定、低漂移、不高频抖脚 | 10 s 水平漂移 `< 0.25 m` |
| **外力推撞扰动** | 施加瞬时速度冲量后能在 1–2 s 内恢复平衡 | 机身高度与竖直分量不跌破摔倒阈值 |

运行本仓库内置的跨引擎自动化校验：

```bash
npm run check:simulation
```

## 下一步 \{#next}

- 完成跨引擎验证后，进入 [四足 Sim2Real 工程部署指南](/docs/practices/quadruped/sim2real-guide/placeholder) 准备真机标定与安全上机。
- 动手复现两条完整端到端教程：[Go2 MJX → 官方栈 Sim2Sim](/docs/practices/amd/unitree-rl-mjx/sim2sim) · [Flamingo Isaac Lab → MuJoCo Sim2Sim](/docs/practices/humanoid/flamingo-isaaclab/onnx-and-sim2sim)。
