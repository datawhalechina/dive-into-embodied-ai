# Reproduce this run

On a CUDA box (this run: `a40-box`, NVIDIA A40, device 1):

    CUDA_VISIBLE_DEVICES=1 uv run python -m unitree_rl_mjx.train.go2_velocity \
      --seed 2 --out-dir benchmarks/results/train-go2-a40-full-seed2

Environment per `scripts/setup_cuda.sh` (versions in `run.json`). No
hyperparameter overrides; the full published budget trains by default.
Evaluate with:

    uv run python benchmarks/eval_policy.py \
      benchmarks/results/train-go2-a40-full-seed2/params.bin

> 注：本目录只随仓库分发小体积证据（json/jsonl/npz/csv/md）。上文提到的
> 训练 checkpoint（`params.bin`）、完整 DDS 日志（`*.csv.gz`）、视频（`*.mp4`）
> 与已执行 notebook 不在其中，可按本文步骤复现生成。
