# Notebook end-to-end execution evidence

`notebooks/go2_training_demo.ipynb` executed top-to-bottom without
manual intervention on the ROCm box (Radeon AI PRO R9700, gfx1201),
2026-08-10:

    uv run --with jupyter --with matplotlib --with imageio \
      jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=3600 \
      --output-dir nbdemo-out --output executed-demo.ipynb \
      notebooks/go2_training_demo.ipynb

Result: backend `gpu`, 40M-step smoke training in 1761 s wall, final
episode reward 46.3, 251-frame rendered rollout gif — all captured inline
in `executed-demo.ipynb` (curve + gif in cell outputs). `run.json` is the
training run's own manifest.

> 注：本目录只随仓库分发小体积证据（json/jsonl/npz/csv/md）。上文提到的
> 训练 checkpoint（`params.bin`）、完整 DDS 日志（`*.csv.gz`）、视频（`*.mp4`）
> 与已执行 notebook 不在其中，可按本文步骤复现生成。
