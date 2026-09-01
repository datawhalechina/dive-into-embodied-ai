# MicroDuck RL：独立仿真环境

这是 Dive into Embodied AI 对 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的教程集成。项目使用 MuJoCo、mjlab、MuJoCo Warp 和 PPO，为约 800 g 的 MicroDuck 双足机器人训练速度跟踪策略。

## 本目录做什么

- `src/mjlab_microduck/`：机器人 MJCF、网格资产、执行器模型和任务配置。
- `scripts/`：训练后播放、ONNX 导出、CPU 推理和观测对比工具。
- `tests/`：配置不变量和奖励函数回归测试。
- `pyproject.toml` + `uv.lock`：与上游隔离的 Python 3.12 环境。

本次教程范围是“环境搭建 + GPU smoke test”，不包含 4096 并行环境的完整训练，也不包含真机部署。完整上游说明保存在 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

## 创建环境

在本目录执行：

```bash
cd codes/practices/humanoid/microduck-rl
UV_HTTP_TIMEOUT=600 uv sync --locked
```

首次安装会下载 Torch、CUDA runtime、Warp 和 MuJoCo 等大型依赖。需要 NVIDIA GPU；如果机器驱动与 CUDA 组件不匹配，应优先修复驱动或改用 GPU 容器。

## 最小验证

先确认任务注册：

```bash
uv run list-envs | grep MicroDuck
```

再运行 64 个环境、5 个 iteration 的 smoke test：

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

这个检查只验证环境构建、GPU stepping、观测维度、奖励计算和 NaN 防护，不代表步态已经训练成功。完整训练前应先检查显存和每轮耗时。

在 Driver 535 / 系统 CUDA 12.2 的 x86_64 Linux 机器上，使用兼容分支的 Torch 组合：

```text
torch==2.7.1+cu126
warp-lang==1.12.0
mjlab==1.3.0
mujoco-warp==3.8.1
```

该组合已在 RTX 3050 Laptop 4 GiB 上完成 5/5 iteration，退出码为 0；每轮耗时 4.40s、4.21s、3.85s、4.03s、4.23s，生成 `model_4.pt` 与 ONNX 文件。Driver 535 下 CUDA Graphs 会被禁用，但普通 GPU stepping 不受影响。

有桌面显示会话时，可以用最新 smoke checkpoint 打开 viewer：

```bash
RUN_DIR="$(find logs/rsl_rl/velocity -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "$RUN_DIR/model_4.pt" \
  --num-envs 1
```

这只是 checkpoint 加载和推理链路 demo；5 iteration 不代表策略已经收敛为稳定步态。

CPU 侧的配置与奖励回归测试：

```bash
uv run --with pytest pytest tests/ -q
```

兼容分支最近一次结果：`154 passed, 1 skipped`。

## 上游与许可证

当前集成基于上游 `develop` 分支 commit `d424a0c899f6b33cbd3daeb279913134349c0b63`。代码按上游 Apache-2.0 许可证保留；3D 模型文件按上游说明使用 CC BY-SA-NC，不能脱离相应署名和非商业条款单独再授权。

上游项目：<https://github.com/pollen-robotics/microduck_rl>
