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
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

这个检查只验证环境构建、GPU stepping、观测维度、奖励计算和 NaN 防护，不代表步态已经训练成功。完整训练前应先检查显存和每轮耗时。

## 上游与许可证

当前集成基于上游 `develop` 分支 commit `d424a0c899f6b33cbd3daeb279913134349c0b63`。代码按上游 Apache-2.0 许可证保留；3D 模型文件按上游说明使用 CC BY-SA-NC，不能脱离相应署名和非商业条款单独再授权。

上游项目：<https://github.com/pollen-robotics/microduck_rl>
