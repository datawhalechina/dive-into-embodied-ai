"""从 TensorBoard 与 benchmark JSON 生成训练结果图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)


def _series(accumulator, tag):
    values = accumulator.Scalars(tag)
    return (
        np.asarray([value.step for value in values], dtype=float),
        np.asarray([value.value for value in values], dtype=float),
    )


def _smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smoothed])


def plot_training(run_dir, output):
    event_path = next((run_dir / "tb" / "PPO_1").glob("events.*"))
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    for tag, label, color in (
        ("rollout/ep_rew_mean", "episode return", "tab:blue"),
        ("rollout/ep_len_mean", "episode length", "tab:green"),
    ):
        steps, values = _series(accumulator, tag)
        axis = axes[0, 0] if "rew" in tag else axes[0, 1]
        axis.plot(steps / 1e6, values, color=color, alpha=0.2)
        axis.plot(steps / 1e6, _smooth(values), color=color, label=label)
        axis.set_xlabel("environment steps [M]")
        axis.grid(alpha=0.25)
        axis.legend()

    for tag, label in (
        ("reward_components/r_tracking_lin_vel", "linear tracking"),
        ("reward_components/r_tracking_ang_vel", "yaw tracking"),
    ):
        steps, values = _series(accumulator, tag)
        axes[1, 0].plot(steps / 1e6, _smooth(values), label=label)
    axes[1, 0].set_xlabel("environment steps [M]")
    axes[1, 0].set_ylabel("scaled reward component")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    steps, kl_values = _series(accumulator, "train/approx_kl")
    axes[1, 1].plot(steps / 1e6, _smooth(kl_values), label="approx KL")
    axes[1, 1].axhline(0.02, color="tab:red", linestyle="--", label="target KL")
    std_steps, std_values = _series(accumulator, "train/std")
    std_axis = axes[1, 1].twinx()
    std_axis.plot(
        std_steps / 1e6,
        _smooth(std_values),
        color="tab:orange",
        label="action std",
    )
    axes[1, 1].set_xlabel("environment steps [M]")
    axes[1, 1].set_ylabel("KL")
    std_axis.set_ylabel("action std")
    axes[1, 1].grid(alpha=0.25)
    lines = axes[1, 1].lines + std_axis.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="best")

    run_name = run_dir.name.replace("_", " ")
    figure.suptitle(f"Pupper PPO ROCm training — {run_name}")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_benchmarks(run_dir, output):
    config_path = run_dir / "training_config.json"
    final_step = 20.0
    if config_path.exists():
        config = json.loads(config_path.read_text())
        final_step = config["timesteps"] / 1e6

    candidates = [
        (1, "benchmark_1000000.json"),
        (2, "benchmark_2000000.json"),
        (5, "benchmark_5000000.json"),
        (10, "benchmark_10000000.json"),
        (15, "benchmark_15000000.json"),
        (final_step, "benchmark_final.json"),
    ]
    rows = []
    for step, filename in candidates:
        path = run_dir / filename
        if path.exists():
            rows.append((step, json.loads(path.read_text())["summary"]))

    steps = [row[0] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, [row[1]["score"] for row in rows], "o-", label="score")
    axes[0].plot(
        steps,
        [100 * row[1]["survival_ratio"] for row in rows],
        "o-",
        label="survival [%]",
    )
    axes[0].set_xlabel("environment steps [M]")
    axes[0].set_ylabel("score / percent")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        steps,
        [row[1]["translation_rmse"] for row in rows],
        "o-",
        label="translation RMSE [m/s]",
    )
    axes[1].plot(
        steps,
        [row[1]["yaw_command_rmse"] for row in rows],
        "o-",
        label="yaw RMSE [rad/s]",
    )
    axes[1].set_xlabel("environment steps [M]")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Fixed-command benchmark progression")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="绘制 Pupper 训练结果")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training(args.run_dir, args.output_dir / "training_curves.png")
    plot_benchmarks(args.run_dir, args.output_dir / "benchmark_progress.png")
    print(f"结果图已保存：{args.output_dir}")


if __name__ == "__main__":
    main()
