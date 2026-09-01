# Reproduce this run

On a ROCm box (this run: `r9700-box`, Radeon AI PRO
R9700, gfx1201). The XLA flag is the documented substrate constraint, not a
configuration fork:

    XLA_FLAGS=--xla_gpu_enable_command_buffer= \
      uv run python -m unitree_rl_mjx.train.go2_velocity \
      --seed 2 --out-dir benchmarks/results/train-go2-r9700-full-seed2

Environment per `scripts/setup_rocm.sh` (versions in `run.json`). No
hyperparameter overrides; the full published budget trains by default.
Evaluate with:

    uv run python benchmarks/eval_policy.py \
      benchmarks/results/train-go2-r9700-full-seed2/params.bin

> 注：本目录只随仓库分发小体积证据（json/jsonl/npz/csv/md）。上文提到的
> 训练 checkpoint（`params.bin`）、完整 DDS 日志（`*.csv.gz`）、视频（`*.mp4`）
> 与已执行 notebook 不在其中，可按本文步骤复现生成。
