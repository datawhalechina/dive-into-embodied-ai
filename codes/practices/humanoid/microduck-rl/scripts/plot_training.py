#!/usr/bin/env python3
"""Render a compact training report from an mjlab TensorBoard run.

The trainer writes one scalar event file per run.  This script keeps the
visualization reproducible and deliberately reads only public scalar metrics;
checkpoints and W&B artifacts are not required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all scalar series from *run_dir*, keyed by TensorBoard tag."""
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no TensorBoard event file found in {run_dir}")

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        series[tag] = (
            np.asarray([item.step for item in values], dtype=np.int64),
            np.asarray([item.value for item in values], dtype=np.float64),
        )
    return series


def _plot_series(ax, series, tag: str, label: str | None = None, smooth: int = 1):
    """Plot one scalar if it exists and return whether it was available."""
    if tag not in series:
        return False
    steps, values = series[tag]
    ax.plot(steps, values, alpha=0.25, linewidth=0.8)
    if smooth > 1 and len(values) >= smooth:
        kernel = np.ones(smooth, dtype=np.float64) / smooth
        smoothed = np.convolve(values, kernel, mode="valid")
        ax.plot(steps[smooth - 1 :], smoothed, linewidth=1.8, label=label or tag)
    else:
        ax.plot(steps, values, linewidth=1.8, label=label or tag)
    return True


def make_report(run_dir: Path, output: Path, smooth: int) -> dict[str, float]:
    series = _load_scalars(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    fig.suptitle(f"MicroDuck RL training report\n{run_dir.name}", fontsize=14)

    reward_ax, length_ax, error_ax, failure_ax = axes.flat
    _plot_series(reward_ax, series, "Train/mean_reward", "mean reward", smooth)
    reward_ax.set_title("Learning signal")
    reward_ax.set_xlabel("iteration")
    reward_ax.set_ylabel("mean reward")
    reward_ax.grid(alpha=0.25)
    reward_ax.legend(loc="best")

    _plot_series(length_ax, series, "Train/mean_episode_length", "episode length", smooth)
    length_ax.set_title("Episode survival")
    length_ax.set_xlabel("iteration")
    length_ax.set_ylabel("steps")
    length_ax.grid(alpha=0.25)
    length_ax.legend(loc="best")

    has_xy = _plot_series(error_ax, series, "Metrics/twist/error_vel_xy", "xy error", smooth)
    has_yaw = _plot_series(error_ax, series, "Metrics/twist/error_vel_yaw", "yaw error", smooth)
    error_ax.set_title("Command tracking error")
    error_ax.set_xlabel("iteration")
    error_ax.set_ylabel("error")
    error_ax.grid(alpha=0.25)
    if has_xy or has_yaw:
        error_ax.legend(loc="best")

    has_fall = _plot_series(failure_ax, series, "Episode_Termination/fell_over", "fell over", smooth)
    has_timeout = _plot_series(failure_ax, series, "Episode_Termination/time_out", "time out", smooth)
    failure_ax.set_title("Termination mix")
    failure_ax.set_xlabel("iteration")
    failure_ax.set_ylabel("episodes")
    failure_ax.grid(alpha=0.25)
    if has_fall or has_timeout:
        failure_ax.legend(loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    summary: dict[str, float] = {}
    if "Train/mean_reward" in series:
        _, values = series["Train/mean_reward"]
        summary["initial_mean_reward"] = float(values[0])
        summary["final_mean_reward"] = float(values[-1])
        summary["best_mean_reward"] = float(values.max())
    if "Train/mean_episode_length" in series:
        _, values = series["Train/mean_episode_length"]
        summary["final_mean_episode_length"] = float(values[-1])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="mjlab run directory")
    parser.add_argument("--out", type=Path, required=True, help="image output path (WebP recommended)")
    parser.add_argument("--smooth", type=int, default=15, help="moving-average window")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be >= 1")
    summary = make_report(args.run_dir, args.out, args.smooth)
    print(f"saved {args.out}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
