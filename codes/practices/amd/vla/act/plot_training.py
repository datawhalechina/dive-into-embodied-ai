"""把 ACT 的 CSV 训练日志绘制成教程用曲线图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS = LAB_DIR / "outputs" / "act_aloha_transfer_rocm_full" / "training_metrics.csv"
DEFAULT_REPORT_DIR = LAB_DIR / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 AMD ROCm ACT 训练曲线")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--ema-span", type=int, default=80)
    args = parser.parse_args()
    if args.ema_span < 1:
        parser.error("--ema-span 必须大于 0")
    return args


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"训练日志不存在：{path}")
    with path.open(encoding="utf-8", newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    if not rows:
        raise ValueError(f"训练日志没有数据：{path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#475569")


def plot_metrics(metrics: dict[str, np.ndarray], output_dir: Path, ema_span: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    step = metrics["step"]
    loss = metrics["loss"]
    l1_loss = metrics["l1_loss"]
    kld_loss = metrics["kld_loss"]
    grad_norm = metrics["grad_norm"]
    speed = metrics["steps_per_second"]
    memory = metrics["max_gpu_memory_gb"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.patch.set_facecolor("#f8fafc")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        style_axis(axis)

    axes[0, 0].plot(step, loss, color="#93c5fd", linewidth=0.8, alpha=0.45, label="raw")
    axes[0, 0].plot(step, ema(loss, ema_span), color="#2563eb", linewidth=2.2, label="EMA")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("ACT objective", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("loss (log scale)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(step, ema(l1_loss, ema_span), color="#059669", linewidth=2.0, label="L1")
    axes[0, 1].plot(
        step,
        ema(10.0 * kld_loss, ema_span),
        color="#d97706",
        linewidth=2.0,
        label="10 × KL",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Loss decomposition", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("weighted component (log scale)")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(step, grad_norm, color="#c4b5fd", linewidth=0.8, alpha=0.4)
    axes[1, 0].plot(step, ema(grad_norm, ema_span), color="#7c3aed", linewidth=2.0)
    axes[1, 0].axhline(10.0, color="#ef4444", linestyle="--", linewidth=1.2, label="clip = 10")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Gradient norm before clipping", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("optimizer step")
    axes[1, 0].set_ylabel("L2 norm (log scale)")
    axes[1, 0].legend(frameon=False)

    speed_axis = axes[1, 1]
    speed_axis.plot(step, speed, color="#0f766e", linewidth=1.8, label="step/s")
    speed_axis.set_title("ROCm throughput and memory", loc="left", fontweight="bold")
    speed_axis.set_xlabel("optimizer step")
    speed_axis.set_ylabel("steps / second", color="#0f766e")
    memory_axis = speed_axis.twinx()
    memory_axis.plot(step, memory, color="#e11d48", linewidth=1.5, alpha=0.8, label="VRAM")
    memory_axis.set_ylabel("peak VRAM (GiB)", color="#e11d48")
    memory_axis.spines["top"].set_visible(False)

    elapsed_hours = metrics["elapsed_seconds"][-1] / 3600.0
    figure.suptitle(
        "LeRobot ACT · ALOHA Transfer Cube · AMD ROCm\n"
        f"{int(step[-1]):,} steps · {elapsed_hours:.2f} h · BF16 / batch 32",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    webp_path = output_dir / "training_curves.webp"
    figure.savefig(webp_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)

    tail_start = max(int(len(step) * 0.8), 0)
    summary = {
        "steps": int(step[-1]),
        "recorded_points": len(step),
        "initial_loss": float(loss[0]),
        "final_loss": float(loss[-1]),
        "best_loss": float(np.min(loss)),
        "final_l1_loss": float(l1_loss[-1]),
        "final_kld_loss": float(kld_loss[-1]),
        "elapsed_hours": elapsed_hours,
        "median_steps_per_second_last_20_percent": float(np.median(speed[tail_start:])),
        "peak_gpu_memory_gb": float(np.max(memory)),
        "ema_span": ema_span,
    }
    (output_dir / "training_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"训练曲线：{webp_path}")
    return summary


def main() -> None:
    args = parse_args()
    plot_metrics(load_metrics(args.metrics), args.output_dir, args.ema_span)


if __name__ == "__main__":
    main()
