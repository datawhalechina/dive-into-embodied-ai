# MicroDuck RL：独立仿真环境

这是 Dive into Embodied AI 对 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的教程集成。项目使用 MuJoCo、mjlab、MuJoCo Warp 和 PPO，为约 800 g 的 MicroDuck 双足机器人训练走路、起身、拾取、踢球、轮式移动和翻滚等 18 个动作模式。

教程页面在 [`docs/practices/humanoid/microduck-rl`](../../../../docs/practices/humanoid/microduck-rl/)，里面有动作 GIF / MP4、训练流程、奖励设计、完整模式表和复现命令。本 README 只保留代码目录的快速入口。

## 目录

- `src/mjlab_microduck/`：机器人 MJCF、STL 网格、BAM 执行器和任务配置。
- `scripts/`：训练、续训、离屏渲染、评估和汇总工具。
- `tests/`：配置不变量、奖励函数和数值防护测试。
- `pyproject.toml` + `uv.lock`：独立的 Python 3.12 环境。

## 创建环境

```bash
cd codes/practices/humanoid/microduck-rl
UV_HTTP_TIMEOUT=600 uv sync --locked
```

首次同步会下载 Torch、CUDA runtime、Warp 和 MuJoCo 等较大的 wheel。需要 NVIDIA GPU；如果驱动与用户态 CUDA 组件不匹配，先跑下面的 smoke test确认问题范围。

本次训练使用的兼容组合是：

```text
torch==2.7.1+cu126
warp-lang==1.12.0
mjlab==1.3.0
mujoco-warp==3.8.1
```

它在 Driver 535 / 系统 CUDA 12.2 的机器上可用。Driver 535 下 CUDA Graphs 会被禁用，这是性能降级，不影响普通 GPU stepping。

## 最小验证

```bash
uv run list-envs | grep MicroDuck

WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --env.seed 42 \
  --agent.seed 42 \
  --agent.max-iterations 5
```

5 个 iteration 只验证任务注册、MJCF / 网格加载、GPU stepping、观测奖励计算和 NaN / OOM 防护；它不代表步态已经收敛。

CPU 侧回归测试：

```bash
uv run --with pytest pytest tests/ -q
```

## 18 个模式的训练和评估

项目提供两个串行阶段脚本。它们从各模式的 smoke checkpoint 续训，按任务设定的里程碑保存 checkpoint；串行执行是为了控制显存和日志数量，**每个模式仍然拥有自己的 PPO 网络权重**。

```bash
./scripts/train_stable_matrix_stage1.sh
./scripts/train_stable_matrix_stage2.sh
./scripts/evaluate_stable_matrix.sh
uv run python scripts/summarize_training_matrix.py
```

评估脚本会从每个任务的最后 checkpoint 渲染 200 帧 MP4 / GIF，并把回放统计写入 `logs/mode_matrix_stable/videos/evaluation_status.tsv`。训练日志和模型文件保留在本地，不提交到教程仓库；教程只挑选必要的可视化素材。

本次正式训练在 `dev1-docker` 的 8 × RTX 4090 上完成，18/18 个任务生成 checkpoint，18/18 个任务完成离屏回放。结果边界、模式清单和视频入口请以教程页面为准：

<https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/humanoid/microduck-rl>

## 单个 checkpoint 回放

有显示会话时：

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/mode_matrix_stable/<run>/model_3000.pt \
  --num-envs 1
```

无 `DISPLAY` 时，使用实际 mjlab / BAM 环境离屏渲染：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run python scripts/render_mjlab_checkpoint.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint logs/rsl_rl/mode_matrix_stable/<run>/model_3000.pt \
  --mp4 /tmp/microduck-velocity.mp4 \
  --gif /tmp/microduck-velocity.gif \
  --frames 200 --lin-vel-x 0.15
```

视频是 checkpoint 的评估结果，不是训练真值。要迁移到真机，还需要多 seed、更多速度和地形范围、舵机延迟与摩擦校准、限位保护以及低速 sim2real 测试。

## 上游与许可证

当前集成基于上游 `develop` 分支 commit `d424a0c899f6b33cbd3daeb279913134349c0b63`。代码按 Apache-2.0 许可证保留；3D 模型文件按上游说明使用 CC BY-SA-NC，不能脱离相应署名和非商业条款单独再授权。
