---
title: AUP Learning Cloud 云算力
sidebar_position: 2
displayed_sidebar: practicesAmdSidebar
description: "使用 AUP Learning Cloud 在浏览器中体验基于 AMD Ryzen AI APU 的 JupyterHub、Code Server 与 ROCm 开发环境。"
---

# AUP Learning Cloud 云算力

:::warning[内测说明]

AUP Learning Cloud 目前处于内测阶段，开放范围、硬件配置、会话时长、存储空间与积分规则都可能调整。本文用于帮助你认识和快速上手平台，实际可用资源请以登录后的页面与官方通知为准。

:::

[AUP Learning Cloud](https://tpe.aupcloud.io/) 是 AMD Research 开源的 AI 学习云平台。它把运行在 AMD 硬件上的 **JupyterHub**、**Code Server** 与课程环境放到浏览器中，让没有本地 AMD 设备的学习者也能体验 ROCm、HIP、深度学习和物理仿真实验。

> [立即登录 AUP Learning Cloud](https://tpe.aupcloud.io/) · [查看开源项目](https://github.com/AMDResearch/aup-learning-cloud) · [查看官方部署文档](https://amdresearch.github.io/aup-learning-cloud/)

## 平台有什么不同

AUP Learning Cloud 的主要算力来自 **Ryzen™ AI APU**。CPU、集成 GPU（iGPU）与 NPU 位于同一颗芯片中，比独立显卡云主机更接近 AMD AI PC 的端侧异构计算环境。

参考环境包括以下两类设备，具体可用机型以平台资源选择页为准：

| 设备 | CPU | GPU（iGPU） | NPU |
|---|---|---|---|
| Ryzen™ AI Max+ 395（Strix Halo） | Zen 5，16 核 | Radeon™ 8060S，RDNA 3.5 | XDNA™ 2，约 50 TOPS |
| Ryzen™ AI 9 HX 370（Strix Point） | Zen 5，12 核 | Radeon™ 890M，RDNA 3.5 | XDNA™ 2，约 50 TOPS |

其中，CPU 适合编译、数据预处理与常规计算；GPU 可通过 ROCm 加速 PyTorch 等工作负载；NPU 面向低功耗端侧推理。平台还提供计算机视觉、深度学习、大语言模型、HIP 编程与物理仿真等课程或开发镜像。

## AUP Learning Cloud 与 Radeon Cloud

两者都是免去本地环境配置的 AMD 云端开发入口，但硬件定位和使用方式不同：

| 对比项 | AUP Learning Cloud | AMD Radeon Cloud |
|---|---|---|
| 主要定位 | 教学实验与端侧异构计算体验 | 通用 ROCm 开发与独立 GPU 算力 |
| 典型硬件 | Ryzen AI APU，集成 CPU / GPU / NPU | Radeon PRO W7900D 等专业独立 GPU |
| 开发入口 | JupyterHub、Jupyter Notebook、Code Server | 技术模板与云端工作区 |
| 适合任务 | 课程练习、HIP/ROCm 验证、端侧推理、小规模实验 | 大模型部署与微调、算子优化、计算量更大的训练任务 |

如果你想体验 Ryzen AI 平台上的端侧 GPU/NPU，优先选择 AUP Learning Cloud；如果任务更依赖显存容量或长时间独立 GPU 计算，可以进一步了解 [AMD Radeon Cloud 指南](https://github.com/datawhalechina/hello-rocm/blob/master/docs/zh/cloud/amd-radeon-cloud.md)。

## 快速开始

### 1. 登录平台

打开 [tpe.aupcloud.io](https://tpe.aupcloud.io/)，选择以下任一方式登录：

- **GitHub 登录**：点击 **Sign in with GitHub** 完成授权；
- **本地账户**：使用管理员提供的用户名和密码，首次登录时按提示修改密码。

能否直接进入资源选择页取决于当前内测开放范围与账号权限。

### 2. 选择工作环境

根据任务选择入口：

- **Jupyter Notebook**：适合跟随课程、运行 Notebook 和快速验证代码；
- **Code Server GPU Environment**：适合克隆完整项目、使用终端、调试 Python 或 HIP 程序；
- **Code Server CPU Environment**：适合阅读代码、数据处理和不需要 GPU 的开发任务。

选择镜像与硬件后，设置运行时长并启动服务器。GPU 资源通常消耗更多积分，不需要 GPU 时应优先选择 CPU 环境。

### 3. 验证 ROCm 与 PyTorch

进入 GPU 环境后，在终端执行：

```bash
rocm-smi
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

ROCm 版本的 PyTorch 仍使用 `torch.cuda` 接口判断和调用 GPU，这是兼容接口的正常行为。

### 4. 保存项目文件

参考平台当前规则，每位用户默认拥有 **20 GB** 持久化空间。镜像的默认工作目录可能随会话重置，重要代码、Notebook 和实验结果应在会话结束前保存到 `/home/jovyan`：

```bash
cd /home/jovyan
git clone https://github.com/datawhalechina/dive-into-embodied-ai.git
```

### 5. 正确停止服务器

实验完成后不要只关闭浏览器标签页。进入 **File → Hub Control Panel**，点击 **Stop My Server** 释放资源；使用 Code Server 时，也可以先通过左下角的 JupyterHub 入口返回控制面板。

## 用于具身智能学习

AUP Learning Cloud 适合先完成环境验证和小规模实验，再决定是否迁移到本地工作站或独立 GPU 云平台：

1. 在 GPU 环境中确认 ROCm、PyTorch 与设备均可见；
2. 运行物理仿真或机器人学习项目的最小示例；
3. 记录软件版本、设备名称、运行时间和显存占用；
4. 再尝试 AMD 专区的 [Pupper Locomotion](./pupper-control/locomotion) 或 [Pupper VLA](./pupper-control/vla) 项目。

:::tip[控制实验规模]

平台当前参考规则中，单次容器默认运行时长为 20 分钟，且按“运行分钟数 × 硬件倍率”消耗积分。因此它更适合环境验证、推理和短实验；完整强化学习训练应先确认可申请的会话时长与额度。

:::

## 常见问题

**为什么已经启动一个环境，却无法切换到另一个？** 先在 Hub 控制面板停止当前服务器，再选择新镜像。

**会话结束后文件不见了怎么办？** 只有持久化目录中的文件会被保留。请把重要内容复制到 `/home/jovyan`，并在结束前确认文件已经写入。

**Code Server 适合做什么？** 它提供接近桌面版 VS Code 的编辑、终端、调试、扩展与端口转发体验，适合开发完整项目；一次性的 Notebook 实验则使用 JupyterHub 更直接。

## 参考资料

- [hello-rocm：AUP Learning Cloud 使用指南](https://github.com/datawhalechina/hello-rocm/blob/master/docs/zh/cloud/aup-learning-cloud.md)
- [hello-rocm：AMD Radeon Cloud 云算力](https://github.com/datawhalechina/hello-rocm/blob/master/docs/zh/cloud/amd-radeon-cloud.md)
- [AMDResearch/aup-learning-cloud](https://github.com/AMDResearch/aup-learning-cloud)
