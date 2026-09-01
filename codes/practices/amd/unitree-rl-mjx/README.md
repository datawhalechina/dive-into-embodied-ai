# unitree-rl-mjx

宇树官方 RL 训练套件 `unitree_rl_mjlab`（MuJoCo-Warp，CUDA-only）的 MJX/JAX 移植，
同一份代码在 AMD（ROCm）与 NVIDIA（CUDA）GPU 上训练，导出的 ONNX 策略直接进
官方未改动的 C++ 部署 / sim2sim 栈。

当前覆盖：**Go2 · 平地速度跟踪**（官方套件 7 机器人 × 2 任务族中的一个任务）。
移植钉在参考仓 `unitree_rl_mjlab` 提交 `1425b15f`，逐奖励项对齐。

## 快速开始

```bash
scripts/fetch_go2_meshes.sh   # 从上游钉住的提交拉取 Go2 可视网格（带校验，仅首次需要）
scripts/setup_rocm.sh   # AMD GPU（gfx1201 / R9700 上验证）
scripts/setup_cuda.sh   # NVIDIA GPU
scripts/setup_cpu.sh    # CPU（开发用）
uv run pytest           # CPU 即可运行；依赖 checkpoint 的用例自动跳过
uv run python -m unitree_rl_mjx.train.go2_velocity --seed 0
```

每个 setup 脚本会写出 `env-manifest.txt` 记录实际安装的工具链版本。Go2 的 16 个
`.obj` 可视网格（约 27MB）不随仓库分发，由 `fetch_go2_meshes.sh` 按 SHA-256 校验
从 `unitree_rl_mjlab@1425b15f` 拉取——MJCF 引用了它们，取回之前环境无法加载。

## 目录

```
src/unitree_rl_mjx/    库本体（环境、任务配置、训练入口、策略导出）
tests/                 pytest 套件（CPU-only）
benchmarks/            吞吐 / 收敛 / sim2sim 度量 harness
benchmarks/results/    已发布运行的元数据与指标（json/jsonl/npz/csv + REPRODUCE.md）
policies/go2/velocity/ 训练好的策略（ONNX + deploy.yaml），可直接装进官方部署栈
docs/tutorials/        环境搭建 → 训练 → sim2sim 三篇教程
notebooks/             端到端演示 notebook
```

## 说明

- `benchmarks/results/` 只含小体积证据文件；训练 checkpoint（`params.bin`）、
  DDS 全量日志与视频未随仓库分发，因此 `tests/test_policy_export.py` 中依赖
  checkpoint 的用例会自动跳过，其余测试不受影响。
- 基准数字所在机器以角色标签标注（`a40-box` / `r9700-box` / `m4-laptop`），
  每次运行的 `run.json` 记录设备型号与软件版本。
- 本目录代码与资产按 [LICENSE](LICENSE)（Apache-2.0）分发；Go2 模型资产的
  来源与许可见 `src/unitree_rl_mjx/assets/robots/unitree_go2/PROVENANCE.md`。
