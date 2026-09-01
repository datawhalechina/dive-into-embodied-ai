# MicroDuck RL：独立仿真环境

这是 Dive into Embodied AI 对 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的教程集成。项目使用 MuJoCo、mjlab、MuJoCo Warp 和 PPO，为约 800 g 的 MicroDuck 双足机器人训练速度跟踪策略。

## 本目录做什么

- `src/mjlab_microduck/`：机器人 MJCF、网格资产、执行器模型和任务配置。
- `scripts/`：训练后播放、ONNX 导出、CPU 推理和观测对比工具。
- `tests/`：配置不变量和奖励函数回归测试。
- `pyproject.toml` + `uv.lock`：与上游隔离的 Python 3.12 环境。

基础路径是“环境搭建 + GPU smoke test”；本目录还提供一个 500 iteration 的较长训练和可视化复现实验，但不等同于 4096 并行环境的完整训练，也不包含真机部署。完整上游说明保存在 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

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

## 较长训练与可视化

如果希望观察策略从随机探索到较持续控制的变化，可以在当前 Driver 535 兼容环境中运行 500 iteration：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --env.seed 42 \
  --agent.seed 42 \
  --agent.max-iterations 500 \
  --agent.save-interval 100 \
  --agent.logger tensorboard \
  --agent.experiment-name velocity_long \
  --agent.run-name cuda122-500 \
  --agent.upload-model False
```

训练结束后，用 TensorBoard event 文件生成静态报告：

```bash
RUN_DIR="$(find logs/rsl_rl/velocity_long -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
uv run python scripts/plot_training.py \
  --run-dir "$RUN_DIR" \
  --out "$RUN_DIR/microduck-training-500.webp"
```

脚本绘制 mean reward、episode length、速度命令跟踪误差和终止原因；也可以直接查看本次实测的[训练曲线](../../../../docs/practices/humanoid/microduck-rl/figs/microduck-training-500.webp)和[近景 GIF](../../../../docs/practices/humanoid/microduck-rl/figs/microduck-training-500.gif)，或下载[原始 MP4](../../../../docs/practices/humanoid/microduck-rl/figs/microduck-training-500.mp4)。

本次固定 seed 的结果：初始 mean reward `0.1711`，最终 `2.5424`，最高 `2.7410`；最终 mean episode length `57.69` steps，累计 `768,000` transitions，训练耗时约 16 分钟。该曲线用于展示学习趋势和回归过程，不能替代完整 locomotion 收敛评估。

如果目标是稳定步态，可从 `env256-bench` 的最新 checkpoint 继续训练。当前 4 GiB 开发机上的实测命令如下（本次额外运行 3000 iterations，预计约 2–3 小时）：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 256 \
  --env.seed 42 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run '2026-09-01_16-26-47_env256-bench' \
  --agent.load-checkpoint 'model_518.pt' \
  --agent.max-iterations 3000 \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.experiment-name velocity_long \
  --agent.run-name env256-long \
  --agent.upload-model False
```

不要仅凭 reward 宣称稳定：至少同时检查 episode length、`Episode_Termination/fell_over`、速度跟踪误差，并用固定 checkpoint 做 200 帧以上离屏回放。训练产物继续留在本地 `logs/`，教程只提交可复现命令、曲线和经过验收的媒体。

无显示会话时，可用 CPU ONNX + EGL 离屏回放，避免启动 Viser：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run python scripts/render_checkpoint.py \
  --onnx logs/rsl_rl/velocity_long/<run-name>/<run-name>.onnx \
  --mp4 logs/rsl_rl/velocity_long/<run-name>/checkpoint.mp4 \
  --gif logs/rsl_rl/velocity_long/<run-name>/checkpoint.gif \
  --frames 200 --lin-vel-x 0.15
```

脚本会打印 `fallen_fraction`、`min_trunk_z_m` 和 `min_upright_proxy`。这些是回放验收辅助量，不替代训练环境中的 termination 统计；例如 `fallen_fraction` 较高时，即使 reward 很高，也不应把该 checkpoint 发布成稳定步态。

本次深度训练选用 `model_1500.pt`：从 `model_750.pt` 续训到 iteration 1500，256 envs、固定 seed 42。训练侧最后一轮 mean episode length 为 `760.23` steps，`fell_over=0.125`；实际 mjlab/BAM 离屏回放 200 帧（4 秒）`done_count=0`、`fell_like_fraction=0`，因此可以作为教程展示级稳定步态素材。注意这不是多 seed、rough terrain 或真机验收。

## 上游与许可证

当前集成基于上游 `develop` 分支 commit `d424a0c899f6b33cbd3daeb279913134349c0b63`。代码按上游 Apache-2.0 许可证保留；3D 模型文件按上游说明使用 CC BY-SA-NC，不能脱离相应署名和非商业条款单独再授权。

上游项目：<https://github.com/pollen-robotics/microduck_rl>
