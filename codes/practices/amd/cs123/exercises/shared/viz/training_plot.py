"""从 SB3 tensorboard event 文件抽训练曲线画三栏图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_tb_scalars(events_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """从 tensorboard event 文件里读取指定 tag 的 (steps, values)。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("需要 tensorboard 包来读取训练日志")

    ea = EventAccumulator(str(events_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def plot_training_curves(events_dir: str | Path, out_path: str | Path) -> None:
    """画 ep_rew_mean / clip_fraction / explained_variance 三栏图。"""
    events_dir = Path(events_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tags = [
        ("rollout/ep_rew_mean", "ep_rew_mean", "tab:blue"),
        ("train/clip_fraction", "clip_fraction", "tab:orange"),
        ("train/explained_variance", "explained_variance", "tab:green"),
    ]

    # SB3 SubprocVecEnv sometimes doesn't log ep_rew_mean; fall back to value_loss
    steps_check, _ = _read_tb_scalars(events_dir, tags[0][0])
    if len(steps_check) == 0:
        tags[0] = ("train/value_loss", "value_loss", "tab:blue")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (tag, label, color) in zip(axes, tags):
        steps, values = _read_tb_scalars(events_dir, tag)
        if len(steps) > 0:
            ax.plot(steps, values, color=color, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.3)
    fig.suptitle("PPO Training Curves")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
