# 教程 1：环境安装

`unitree_rl_mjx` 是基于 MuJoCo MJX / JAX 的 Unitree 四足强化学习训练库，
ROCm 优先设计：同一份代码、同一份依赖锁文件，在 AMD 与 NVIDIA GPU 上都能训练，
CPU 也能跑通全部测试。

## 支持矩阵

| 后端 | 已验证硬件 | 说明 |
|---|---|---|
| ROCm | Radeon AI PRO R9700（gfx1201，RDNA4），ROCm 7.2.4 | 完整训练已验证收敛 |
| CUDA | NVIDIA A40 / RTX 3090，CUDA 12 | 完整训练已验证收敛 |
| CPU | 任意 x86_64 / arm64 | 测试与小规模冒烟，不适合完整训练 |

## 本地安装

依赖用 [uv](https://docs.astral.sh/uv/) 管理，Python ≥ 3.12：

```bash
git clone https://github.com/datawhalechina/dive-into-embodied-ai.git
cd dive-into-embodied-ai/codes/practices/amd/unitree-rl-mjx

# 拉取 Go2 可视网格（MJCF 引用了它们；带 SHA-256 校验，仅首次需要）：
bash scripts/fetch_go2_meshes.sh

# ROCm 机器：
bash scripts/setup_rocm.sh

# NVIDIA 机器：
bash scripts/setup_cuda.sh

# 仅 CPU：
bash scripts/setup_cpu.sh
```

安装脚本会把 jax 钉在锁文件版本上再叠加对应的 GPU 插件，
并在仓库根写出 `env-manifest.txt` 记录实际安装的版本。验证：

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
# 期望：gpu [RocmDevice(id=0)] 或 gpu [CudaDevice(id=0)]
```

带核显的 AMD 机器注意：`HIP_VISIBLE_DEVICES=0` 选中独显
（核显 gfx1036 不受支持，安装脚本已默认导出）。

## 云端（AMD AUP 平台）

在 AUP 上选择带 ROCm 的 GPU 实例镜像后，安装步骤与本地完全相同。
`notebooks/go2_training_demo.ipynb` 提供从安装检查到训练出图的一键流程，
适合作为实例可用性的自检。

## gfx1201（RDNA4）已知约束

这些约束都已在真实硬件上定位并有稳定的绕开方式，训练结果经过与 CUDA
的收敛一致性对比验证：

1. **训练必须禁用 XLA 命令缓冲**，否则出现数值 NaN：

   ```bash
   export XLA_FLAGS=--xla_gpu_enable_command_buffer=
   ```

   训练文档与脚本中所有 ROCm 命令都带这条环境变量。

2. **4608–16384 并行区间存在设备端排序缺陷**（上游未修复），
   默认配置的 4096 并行不受影响；自定义并行数时避开该区间，
   或用 ≥ 32768（hipCUB 路径）。

3. **同进程连续大批量重编译可能触发 HIP 队列段错误**；
   基准脚本用「每个批量一个子进程 + 周期排空队列」的方式绕开，
   正常训练流程不受影响。

## 显存需求

完整训练配置（4096 并行环境）实测峰值显存：ROCm 约 4.6 GB，CUDA 约 1.5 GB
（JAX 默认会预分配 75% 显存，属正常现象；`XLA_PYTHON_CLIENT_PREALLOCATE=false`
可改为按需分配）。8 GB 显存的卡即可训练。

下一步：[教程 2：训练 Go2 速度策略](02-training.md)
