#!/usr/bin/env python3
"""Summarize the latest TensorBoard run for each MicroDuck mode.

The script intentionally reports trends rather than inventing a binary
"converged" label.  It records the last-step reward/episode length and the
last-20-iteration average, plus any termination counters available for that
task.  A compact WebP plot makes it easy to spot modes that still need a
longer stage-2 run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _latest_run(root: Path, slug: str) -> Path | None:
    if not root.exists():
        return None
    runs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p.name.endswith(slug + "-stage1") or p.name.endswith(slug + "-stage2"))
        and list(p.glob("events.out.tfevents.*"))
    ]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _scalar(acc: EventAccumulator, tag: str) -> list[float]:
    try:
        return [float(item.value) for item in acc.Scalars(tag)]
    except KeyError:
        return []


def summarize(root: Path, output_csv: Path, output_plot: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    # Stage scripts use these stable slugs.  Keeping the list here makes the
    # output deterministic and excludes staging directories without events.
    slugs = (
        "velocity-flat", "velocity-rough", "velstand-flat", "velstand-rough",
        "standup-flat", "standup-rough", "sitstand-flat", "sitstand-rough",
        "groundpick-flat", "groundpick-rough", "ballkick-flat", "roller-velocity",
        "roller-swizzle", "roller-crouch", "roller-slope", "roller-standup",
        "spin-flat", "roulade-flat",
    )
    for slug in slugs:
        run = _latest_run(root, slug)
        if run is None:
            continue
        acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
        acc.Reload()
        rewards = _scalar(acc, "Train/mean_reward")
        lengths = _scalar(acc, "Train/mean_episode_length")
        if not rewards and not lengths:
            continue
        termination_tags = [t for t in acc.Tags().get("scalars", []) if t.startswith("Episode_Termination/")]
        term_last = {t.rsplit("/", 1)[-1]: _scalar(acc, t)[-1] for t in termination_tags if _scalar(acc, t)}
        rows.append({
            "slug": slug,
            "run": run.name,
            "step": float(max(len(rewards), len(lengths)) - 1),
            "reward_last": rewards[-1] if rewards else float("nan"),
            "reward_last20": float(np.mean(rewards[-20:])) if rewards else float("nan"),
            "episode_length_last": lengths[-1] if lengths else float("nan"),
            "timeout_last": term_last.get("time_out", float("nan")),
            "fell_over_last": term_last.get("fell_over", float("nan")),
            "nan_state_last": term_last.get("nan_state", float("nan")),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["slug"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        labels = [str(row["slug"]).replace("-stage1", "") for row in rows]
        x = np.arange(len(rows))
        fig, axes = plt.subplots(3, 1, figsize=(max(12, len(rows) * 0.7), 9.5), sharex=True, constrained_layout=True)
        axes[0].bar(x, [float(row["reward_last20"]) for row in rows], color="#4c78a8")
        axes[0].set_ylabel("reward (last 20)")
        axes[0].set_title("MicroDuck stable-matrix training summary")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x, [float(row["episode_length_last"]) for row in rows], color="#59a14f")
        axes[1].set_ylabel("episode length")
        axes[1].grid(axis="y", alpha=0.25)
        fell = [float(row["fell_over_last"]) if np.isfinite(float(row["fell_over_last"])) else 0.0 for row in rows]
        nan = [float(row["nan_state_last"]) if np.isfinite(float(row["nan_state_last"])) else 0.0 for row in rows]
        axes[2].bar(x, fell, label="fell over", color="#e15759")
        axes[2].bar(x, nan, bottom=fell, label="NaN state", color="#f28e2b")
        axes[2].set_ylabel("termination counter")
        axes[2].grid(axis="y", alpha=0.25)
        axes[2].legend()
        axes[2].set_xticks(x, labels, rotation=55, ha="right")
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot, dpi=160)
        plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/rsl_rl/mode_matrix_stable"))
    parser.add_argument("--csv", type=Path, default=Path("logs/mode_matrix_stable/summary.csv"))
    parser.add_argument("--plot", type=Path, default=Path("logs/mode_matrix_stable/summary.webp"))
    args = parser.parse_args()
    rows = summarize(args.root, args.csv, args.plot)
    print(f"runs={len(rows)} csv={args.csv}")
    if rows:
        print(f"plot={args.plot}")
        for row in rows:
            print(f"{row['slug']}: step={int(row['step'])} reward20={row['reward_last20']:.3f} episode={row['episode_length_last']:.1f}")


if __name__ == "__main__":
    main()
