# MicroDuck RL｜AMD ROCm

MicroDuck 双足 PPO 的独立 AMD 版本。它保留原项目的 MJCF、BAM 执行器、61D actor observation、奖励与 ONNX 导出链路，并为实验性的 ROCm Warp / MuJoCo Warp 增加必要的 HIP 兼容和物理回归。

CUDA 版本位于 [`codes/practices/humanoid/microduck-rl`](../../humanoid/microduck-rl)，完整教程位于 [`docs/practices/amd/microduck-rl`](../../../../docs/practices/amd/microduck-rl/index.md)。

## 状态

- R9700 / `gfx1201` / ROCm 7.2.4 上，Torch HIP、Warp HIP、任务注册与 PPO smoke train 已通过。
- 动态落球 200 step 回归通过：`final_z=0.049633`、`max_contacts=1`。
- 有效 smoke train：64 env 最终约 1,033 steps/s；256 env 最终约 3,741 steps/s；`nan_state=0`。
- 256 env × 500 iteration 的固定 seed 有效长训已完成：307.2 万 transitions，reward `0.0474 → 5.3664`，`nan_state=0`。
- 从该 checkpoint 再续训 1000 iteration，累计 1500 次 PPO update、921.6 万 transitions；续训 reward `0.7702 → 59.0012`，1000/1000 轮的 `nan_state` 与 `below_ground` 均为 0。
- 固定 `0.15 m/s` 前进指令的实际回放中，抽样 checkpoint 以 `model_750.pt` 最好：4 秒前进 `0.325 m`、平均 `0.080 m/s`、无 termination；最终 `model_1498.pt` 仍稳定，但速度回落到 `0.045 m/s`。

ROCm 分支仍属实验后端。这里的兼容层钉住特定源码 commit，不是对其他 AMD GPU 或未来分支的兼容保证。

## 安装

前置条件：x86_64 Linux、Python 3.12、`uv`、`git`、`rocminfo`、完整 ROCm SDK（含 `hipcc`）。

```bash
cd codes/practices/amd/microduck-rl
./scripts/setup_rocm.sh
```

显式指定 R9700 架构：

```bash
HIP_VISIBLE_DEVICES=0 ./scripts/setup_rocm.sh gfx1201
```

脚本创建 `.venv-rocm` 与 `.rocm-src`，使用以下可复现组合：

```text
ROCm SDK 7.2.4
torch 2.9.1+rocm7.2.1
numpy 1.26.4
warp-lang 1.13.0+rocm.0 @ 6530951390fa
mujoco-warp 3.8.1 @ 9229bb9d1a69
mjlab 1.3.0
BAM @ 62bd8ce12154
```

不要执行 `uv sync`：ROCm Warp 需要先编译原生库，普通依赖解析会安装 CUDA/PyPI 版本。

## ROCm 兼容层

`src/mjlab_microduck/__init__.py` 仅在 ROCm Warp 上执行三项处理：

1. 补上 mjlab 1.3 仍使用的 `warp.context` 兼容别名；
2. 默认使用 eager HIP，并关闭 mjlab 外层 graph capture；
3. 禁用 ROCm MuJoCo Warp 错误的静态 broadphase 缓存，确保运动后才接触的几何体仍会进入碰撞候选，并对躯干穿到地面以下增加防御性 termination。

第三项不能省略。未修复版本会缓存首帧候选对，使小黄鸭保持竖直姿态穿过地面；训练日志仍可能显示 reward 上升、episode 变长和 `nan_state=0`。

单独验证动态接触：

```bash
HIP_VISIBLE_DEVICES=0 \
.venv-rocm/bin/python scripts/check_rocm_contacts.py --device cuda:0
```

## 训练

最小 smoke train：

```bash
HIP_VISIBLE_DEVICES=0 WANDB_MODE=offline \
.venv-rocm/bin/train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --env.seed 42 \
  --agent.seed 42 \
  --agent.max-iterations 5
```

固定 seed 的 500 iteration 训练：

```bash
HIP_VISIBLE_DEVICES=0 WANDB_MODE=offline \
.venv-rocm/bin/train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 256 \
  --env.seed 42 \
  --agent.seed 42 \
  --agent.max-iterations 500 \
  --agent.save-interval 100 \
  --agent.logger tensorboard \
  --agent.experiment-name velocity_rocm_contactfix \
  --agent.run-name r9700-env256-iter500-contactfix \
  --agent.upload-model False
```

训练后用实际 mjlab/BAM 环境做 200 帧验收：

```bash
RUN_DIR="$(find logs/rsl_rl/velocity_rocm_contactfix -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv-rocm/bin/python scripts/render_mjlab_checkpoint.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint "$RUN_DIR/$CHECKPOINT" \
  --mp4 "$RUN_DIR/checkpoint.mp4" \
  --frames 200 --lin-vel-x 0.15 --device cuda:0 --seed 42
```

不要只凭 reward 宣称稳定；同时检查 `done_count`、躯干高度、姿态和跌倒比例。

本次 500 iteration 从零训练耗时 803.0 秒，最终 episode length `93.06`，最终/最高吞吐 `3,991 / 4,301 steps/s`；`model_499.pt` 的平均前向速度仅 `0.0052 m/s`，只是站立 checkpoint。续训 1000 iteration 又耗时 1412.2 秒，最终 episode length `764.85`，最终/最高吞吐 `5,275 / 5,517 steps/s`。固定条件下 checkpoint 表现并不随训练轮数单调上升，完整对照、曲线与回放见 AMD 教程。

## 测试

```bash
.venv-rocm/bin/python -m pytest tests/ -q
.venv-rocm/bin/python -m ruff check \
  src/mjlab_microduck/__init__.py \
  src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py \
  src/mjlab_microduck/train_cli.py \
  scripts/check_rocm_contacts.py \
  scripts/render_mjlab_checkpoint.py \
  tests/test_rocm_compat.py \
  tests/test_rocm_dynamic_contacts.py \
  tests/test_rocm_velocity_guard.py
```

当前 AMD 环境实测为 `152 passed`；Ruff 命令只覆盖本版本新增和修改的文件，上游快照的历史格式问题不在本次迁移范围内。

代码沿用上游 Apache-2.0；3D 模型资产沿用上游 CC BY-SA-NC 条款。上游说明见 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。
