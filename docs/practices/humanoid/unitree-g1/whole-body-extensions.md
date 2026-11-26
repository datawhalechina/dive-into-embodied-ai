---
title: 3. 进阶拓展：AMP 动作模仿、起身与 GR00T 全身控制
sidebar_position: 4
displayed_sidebar: practicesHumanoidSidebar
description: "超越平地行走：真实复盘 Unitree G1 上 AMP 动作模仿的成功与起身失败边界、防空翻 Get-Up 奖励设计原则，以及 NVIDIA GR00T Whole-Body Control（SONIC）分层架构。"
---

# 3. 进阶拓展：AMP 动作模仿、起身与 GR00T 全身控制

跑通 12-DoF 平地行走只是人形机器人的“第一道门槛”。真正走向实用场景，必须回答三个更硬核的问题：**如何让步态更自然（AMP）、摔倒后如何安全自主爬起（Get-Up），以及如何把上层 VLA 视觉语言模型接入 29-DoF 全身运控（GR00T WBC / SONIC）？** 本章结合真实实验记录（含关键阴性结果与避坑经验）展开拆解。

## 1. AMP 动作模仿实录：走得更像人，以及一个隐蔽的“假起身”陷阱 \{#amp-experience}

**AMP（Adversarial Motion Priors）** 在常规任务奖励之外引入一个风格判别器（Discriminator），用人类动捕片段（如 LAFAN 数据集）教策略“什么是自然的人类运动”，大幅减轻手工调步态奖励的负担。

我们在 `Unitree-G1-AMP-Flat`（29-DoF，4096 环境 × 25,000 轮，单卡 H200 约 11 小时）上逐文件忠实复现了 `AMP_mjlab` 配方，得到一组非常值得警惕的**双面实测结论**：

- **成功的一面（优秀的自然步态 + 抗扰恢复）**：策略跑通了完整的训练、ONNX 导出与部署链路，走出自然的直立类人步态，并且在遭受 **300 N 侧向推力** 下蹲到 `0.65 m` 后能够**连续、无重置地重新站稳到 `0.77 m`**。
- **失败的一面（从平躺地面自主爬起始终未收敛）**：在关闭环境自动重置后，从俯卧/仰卧姿态（`root_z = 0.09 m`）或遭受 **560 N 硬推倒地**（`root_z = 0.07 m`）测试，策略 12 秒内仅能挣扎抬到约 `0.20 m`，10.7k 轮与 25k 轮几乎毫无进展，`Episode_Reward/track_root_height` 全程停在 `~0.005`。

:::danger[评测避坑铁律：警惕“RSI 瞬移假起身”！]
许多 AMP 训练配置在 play 模式下保留了 `delay_reset_env_ratio = 1.0` 与 `max_delay_steps = 250`（约 5 秒）。一个趴在地上的机器人在 5 秒后会触发**强制重置**，而参考状态初始化（RSI）会把它直接**瞬移**到动捕起身片段的随机一帧——如果刚好抽到站立帧，在抽帧对比图里就会伪装成“自己爬起来了”！

**正确验收方式**：测试起身前必须关闭延迟重置（例如把 `max_delay_steps` 设为 `1e7`），同时连续记录 `root_z(t)` 曲线与 `done/reset` 计数，确认高度是**连续物理爬升**而非单步跳变。
:::

**为什么原配方没学会起身？** 复盘定位到三个结构性原因：
1. **动捕覆盖过薄**：仅包含单条恢复片段（`fallAndGetUp1_subject1`），无法覆盖真实摔倒后的俯卧、仰卧、侧躺等多模态初始姿态。
2. **起身奖励被行走奖励淹没**：高度恢复项仅对延迟环境生效且权重偏弱。
3. **终止惩罚被延迟管理器抑制**：趴在地上不动几乎不扣分，策略落入“躺平不动”的局部最优。

## 2. 专用自主起身（Get-Up）：如何学出平稳起身而非危险空翻 \{#controlled-getup}

既然通用行走+单一 AMP 片段难以兼顾摔倒爬起，主流前沿工作（如 **HoST**、**HumanUP**、**HiFAR**）均采用**专用 Get-Up 策略 + 多阶段状态机切换**。设计一个“能在真机部署、不猛甩空翻”的起身任务，关键抓手有三点：

| 设计维度 | 典型失控表现 | 工程收敛配方 |
| :--- | :--- | :--- |
| **防空翻/防猛弹正则** | 策略学会一脚猛蹬或高角动量空翻，在真机上瞬间打坏减速器 | 组合施加 `action_acc_l2`、`joint_vel_l2`、`joint_torques_l2` 与电功率惩罚，并用课程从早期辅助外力（`apply_external_force_torque`）逐步退火至零 |
| **多姿态初始分布** | 只会从某一个仰面角度起，换侧躺就卡死 | 初始化均匀覆盖俯卧、仰卧、左右侧躺及随机关节扰动，而非依赖单一动捕起始帧 |
| **落地前的代码核验** | 照搬论文超参却没发现框架里缺少某项内建奖励或 body 名称不匹配 | 上训前先在钉死版本的 `mjlab` 中逐项 `import` 核验所有奖励函数、传感器、`G1_ACTION_SCALE`（非统一常数 `0.25`，而与关节力矩/刚度相关）与真实连杆名 |

关于双足前滚翻与趴地起身的完整课程实践，也可以直接对照站内 [MicroDuck 12 步跳墙与翻滚起身课程](/docs/practices/humanoid/microduck-rl/wall-curriculum) 中的**反向课程破局（midroll 初始化）**技巧。

## 3. 全身控制与 VLA 融合：NVIDIA GR00T Whole-Body Control（SONIC） \{#groot-wbc}

当任务从“走到指定位置”升级为“**走到桌子旁、弯腰拾起红杯子并放到托盘里**”时，29 自由度全身协调与多模态语言理解如何衔接？**NVIDIA GR00T Whole-Body Control** 给出了极具代表性的分层解耦思路：

```mermaid
flowchart LR
    VLA["上层 VLA（如 Isaac-GR00T N1.7）<br/>图像 + 语言指令 → 低频 Latent Action"] --> SONIC["底层全身控制（GEAR-SONIC）<br/>50 Hz 解码 Latent Token → 29-DoF 平衡/行走/伸手"]
    SONIC --> DEPLOY["C++20 + TensorRT / ONNX Runtime<br/>Unitree SDK2 实时部署栈"]
    DEPLOY --> G1["Unitree G1（MuJoCo Sim2Sim / 真机）"]
```

- **核心洞见**：不要让大参数量 VLA 直接高频回归 29 个关节角，而是让 **VLA 负责“做什么”（输出低频 SONIC Latent Action）**，由预训练好的 **SONIC 全身控制器以 50 Hz 负责“怎么稳定地用全身做出来”**。
- **单卡工作站 vs 集群分工（算力现实边界）**：
  - **单卡消费级/工作站 GPU（如 RTX 4080 SUPER 16GB）**：非常适合做 MuJoCo 仿真验证、SONIC C++ 部署栈联调、VR/手柄遥操作数据采集、LeRobot 数据清洗与 VLA 推理客户端调试。
  - **多卡集群（4–64+ GPU）**：承担 SONIC 基础全身控制器的全量预训练与 3B 级 Isaac-GR00T N1.7 的常规微调，再将微调权重接回本地工作站/真机闭环执行。

## 4. 延伸阅读与站内联动 \{#next-steps}

- **动手体验**：[在线仿真实验室（Unitree G1 + 外力推撞）](/simulation)
- **基础平衡与跟踪专题**：[双足动态平衡与抗扰控制](../balance-control/placeholder.md) · [人形全身运动跟踪](../motion-tracking/placeholder.md) · [双足高层任务规划](../task-planning-demo/placeholder.md)
- **同系列实操对照**：[两轮足 Flamingo · Isaac Lab 6 章完整课程](../flamingo-isaaclab/intro) · [MicroDuck RL 12 步跳墙课程](../microduck-rl/wall-curriculum)

---

返回：[← Unitree G1 人形运控总览](./index.md)
