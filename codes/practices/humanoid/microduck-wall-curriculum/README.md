# MicroDuck 跳墙课程：跑 → 跳 → 越墙 → 滚翻 → 起身全链

这是 Dive into Embodied AI 对社区课程仓库 [scchy/microduck-rl-cool-curriculum](https://github.com/scchy/microduck-rl-cool-curriculum)（run15 系列）的集成，与旁边的 [MicroDuck RL 基础教程](../microduck-rl/)共用同一上游 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)，但代码树是独立的：本目录包含跳墙课程所需的全部环境配置、奖励函数、课程脚本和探针工具，不混入基础教程的笑 choreo 等实验分支。

教程页面在 [`docs/practices/humanoid/microduck-rl/wall-curriculum.md`](../../../../docs/practices/humanoid/microduck-rl/wall-curriculum.md)，里面有 12 步课程表、每步 GIF 验收、奖励设计速查和 10 条课程心法。本 README 只保留代码目录的快速入口。

## 课程目标与结果

平地跑步 → 跳跃前扑 → 越过矮墙 → 前滚翻 → 站起来继续跑，一条龙全链。已收官：**5cm 墙全链稳定（reward 79.4）**。12 步最小成功路线 ≈ 36–40 GPU 小时。

## 目录

- `src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py`：翻滚 / 跳墙任务族配置，`DIVEWALL_*` 环境变量总开关都在这里。
- `src/mjlab_microduck/tasks/mdp.py`：全部自定义奖励 / 课程函数（翻滚主链、起跳门、过墙结算、midroll 反向课程等）。
- `src/mjlab_microduck/robot/microduck/wall.xml`：墙场景 MJCF；`microduck_constants.py` 里的 `get_wall_spec()` 负责按 `DIVEWALL_WALL_SINK` 改墙高。
- `scripts/train_phase*.sh`：课程各阶段训练脚本，warm-start 靠脚本内 `LOAD_RUN` / `LOAD_CKPT` 接力，可被课程链主脚本覆盖。
- `scripts/probe_*.py`、`scripts/jump_probe.py`：升墙前的物理探针（跳高临界、apex 余量等）。
- `tests/`：配置不变量与奖励函数回归测试（CPU 可跑）。
- `docs/perf/`：训练速度优化战役笔记（4096 envs、njmax 128 等性能配置的由来）。

## 创建环境

```bash
cd codes/practices/humanoid/microduck-wall-curriculum
UV_HTTP_TIMEOUT=600 uv sync --locked
```

需要 NVIDIA GPU。首次同步会下载 Torch、CUDA runtime、Warp 和 MuJoCo 等较大的 wheel。

## 最小验证

```bash
uv run list-envs | grep MicroDuck

# 课程主任务 smoke test：64 envs × 5 iter
WANDB_MODE=offline uv run train Mjlab-RouladeDiveWall-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5
```

5 个 iteration 只验证任务注册、墙场景加载、GPU stepping 和 NaN / OOM 防护，不代表学会跳墙。

CPU 侧回归测试：

```bash
uv run --with pytest pytest tests/ -q
```

## 跑课程

课程表与每步的验收标准见教程页面。通用模式（以第 12 步 B24 为例）：

```bash
./scripts/train_phaseB24_wall_school.sh
```

脚本会 `cd` 到本目录、按当前阶段设置全套 `DIVEWALL_*` 环境变量、从上一个阶段的 checkpoint 续训，并在训练结束后自动渲染回放、转出 GIF。要换 warm 起点或迭代数：

```bash
LOAD_RUN=<run_dir> LOAD_CKPT=<model_XXXX.pt> MAX_ITER=8000 ./scripts/train_phaseB24_wall_school.sh
```

每升一档墙高前，先跑对应的 `scripts/probe_*.py` 探针确认物理余量，再动配置。

## 上游与许可证

- 上游：`scchy/microduck-rl-cool-curriculum`（从 `pollen-robotics/microduck_rl` 的 `d424a0c` 分叉）；集成基线为课程收官 commit `59a4b6d`。
- 代码 Apache-2.0；3D 模型文件按上游说明使用 CC BY-SA-NC，复用时分别检查对应许可。
