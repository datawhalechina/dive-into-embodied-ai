---
title: 1. 环境安装与硬件约束
sidebar_position: 2
displayed_sidebar: practicesAmdSidebar
description: "配置 unitree_rl_mjx 的 ROCm / CUDA / CPU 训练环境，校验拉取 Go2 网格，并了解 RDNA4 (gfx1201) 已知约束与显存需求。"
---

# 1. 环境安装与硬件约束

`unitree_rl_mjx` 是基于 MuJoCo MJX / JAX 的 Unitree 四足强化学习训练库，采用 ROCm 优先设计：同一份代码、同一份依赖锁文件，在 AMD 与 NVIDIA GPU 上都能训练，CPU 也能跑通全部单元测试。

## 支持矩阵 \{#support-matrix}

| 后端 | 已验证硬件 | 说明 |
| :--- | :--- | :--- |
| **ROCm 7.2.4** | Radeon AI PRO R9700（`gfx1201`，RDNA4），Ubuntu 24.04 | 完整 9.8 亿步训练已验证收敛（使用 `jax-rocm7-plugin==0.11.0`） |
| **ROCm 10.1 / 10.2** | Radeon AI PRO R9700（`gfx1201`，RDNA4），Ubuntu 26.04 | 使用 AMD `whl-next` 源的 `jax==0.11.1+rocm10.1.0` 或 `jax==0.11.2+rocm10.2.0`，已实测通过全量单测、GPU 训练与 ONNX 导出对齐 |
| **CUDA** | NVIDIA A40 / RTX 3090，CUDA 12 | 完整训练已验证收敛 |
| **CPU** | 任意 x86_64 / arm64 | 单元测试与小规模冒烟，不适合完整训练 |

## 本地安装 \{#local-install}

依赖使用 [uv](https://docs.astral.sh/uv/) 管理，要求 Python ≥ 3.12：

```bash
git clone https://github.com/datawhalechina/dive-into-embodied-ai.git
cd dive-into-embodied-ai/codes/practices/amd/unitree-rl-mjx

# 拉取 Go2 可视网格（MJCF 引用了 16 个 .obj 网格，约 27 MB，带 SHA-256 校验，仅首次需要）：
bash scripts/fetch_go2_meshes.sh

# ROCm 机器：
bash scripts/setup_rocm.sh

# NVIDIA 机器：
bash scripts/setup_cuda.sh

# 仅 CPU：
bash scripts/setup_cpu.sh
```

安装脚本会将 `jax` 钉在锁文件版本上再叠加对应的 GPU 插件，并在项目根目录写出 `env-manifest.txt` 记录实际安装的工具链版本。验证安装与单元测试：

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
# 期望输出：gpu [RocmDevice(id=0)] 或 gpu [CudaDevice(id=0)]

uv run pytest
```

:::note[ROCm 10.1 / 10.2（Ubuntu 26.04）安装说明]
若宿主机已升级至 **ROCm 10.1 / 10.2**（如 `amdrocm-core10.1-gfx1201` 或 `amdrocm-core10.2-gfx1201`），系统用 `libamd_smi.so` 替代了旧版 `librocm_smi64.so.1`，因此 PyPI 上的 `jax-rocm7-plugin` 会回落至 CPU。此时请直接安装 AMD `whl-next` 源的原生 ROCm 10 插件（注意需要 **`jax >= 0.11.1`** 以包含 MJX Cholesky 所需的 `hipsolver_potrf_ffi`）：

```bash
# ROCm 10.2（推荐，Ubuntu 26.04 源提供 amdrocm-core10.2-gfx1201）：
export LD_LIBRARY_PATH="/opt/rocm/core-10.2/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=0
uv pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "jax==0.11.2" "jaxlib==0.11.2" \
  "jax-rocm10-plugin==0.11.2+rocm10.2.0" \
  "jax-rocm10-pjrt==0.11.2+rocm10.2.0"

# 若宿主机停留在 ROCm 10.1，请对应替换为 core-10.1 与 0.11.1+rocm10.1.0。
```

在 Radeon AI PRO R9700（Ubuntu 26.04.1 LTS）上实测：
- **ROCm 10.1.0 + `jax==0.11.1`**：`--num-timesteps 20971520`（4096 并行环境，实际执行 22,413,312 步）耗时 **835.7 s**（吞吐约 **58.3k steps/s**），评估回报由 `-10.65` 收敛至 **`39.80`**（平均回合长度 `972.5 / 1000`），导出的 ONNX 策略与 JAX 逐元素最大误差 **`1.13e-6`**，20 s 评估轨迹无摔倒行走 **7.77 m**（`up_z` 均值 `0.9965`）。
- **ROCm 10.2.0 + `jax==0.11.2`**：`amdrocm-core10.2-gfx1201` 与 `core-10.1` 可通过 `/etc/alternatives` 共存，原生通过 `Cholesky`、`Go2VelocityFlat` GPU 步进与全量单元测试（`64 passed, 5 skipped`）。完整 **9.84 亿步**训练实测耗时 **`16,136 s`（约 4.48 小时，吞吐 62.4k steps/s）**，比 ROCm 7.2.4（7.9 小时）提速 **1.76×**；最终回报 **`51.20`**（回合长度 `1000 / 1000`），导出的 ONNX 与 JAX 逐元素最大误差 **`1.91e-6`**（评估轨迹 `up_z` 均值 `0.9995`）。
:::

:::note[带核显的 AMD 机器]
使用 `HIP_VISIBLE_DEVICES=0` 选中独显（核显 `gfx1036` 不受支持，安装脚本已默认导出）。
:::

## 云端（AMD AUP 平台） \{#aup-cloud}

在 [AUP Learning Cloud](../aup-learning-cloud.md) 上选择带 ROCm 的 GPU 实例镜像后，安装步骤与本地完全相同。仓库中的 `notebooks/go2_training_demo.ipynb` 提供从安装检查到训练出图的一键流程，适合作为实例可用性的自检。

## `gfx1201`（RDNA4）已知约束 \{#rdna4-constraints}

以下约束均已在真实硬件上定位并具备稳定的绕开方式，训练结果经过与 CUDA 的收敛一致性对比验证：

1. **训练必须禁用 XLA 命令缓冲**，否则会出现数值 `NaN`：

   ```bash
   export XLA_FLAGS=--xla_gpu_enable_command_buffer=
   ```

   教程与脚本中所有 ROCm 训练命令均附带此环境变量。

2. **4608–16384 并行区间存在设备端排序缺陷**（上游未修复）：默认配置的 `4096` 并行不受影响；自定义并行数时应避开该区间，或使用 `≥ 32768`（`hipCUB` 路径）。

3. **同进程连续大批量重编译可能触发 HIP 队列段错误**：基准脚本采用「每个批量一个子进程 + 周期排空队列」绕开，正常单次训练流程不受影响。

## 显存需求 \{#vram}

完整训练配置（4096 并行环境）实测峰值显存：**ROCm 约 4.6 GB，CUDA 约 1.5 GB**。JAX 默认会预分配 75% 显存，属正常现象；设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可改为按需分配。**8 GB 显存的 GPU 即可完整训练。**

---

下一步：[2. 训练 Go2 速度策略 →](./training.md)
