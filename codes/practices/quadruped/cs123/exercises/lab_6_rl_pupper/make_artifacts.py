"""一键：train → command GIF → comparison GIF → 画图 → 写 portfolio。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.viz import gif_utils  # noqa: E402
from starter import (  # noqa: E402
    GIF_FPS,
    GIF_WIDTH,
    PORTFOLIO_DIR,
    render_command_demo,
    render_comparison_gif,
    render_velocity_tracking,
    save_reward_curve,
    save_velocity_tracking,
    train_ppo,
)


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1/5: PPO 训练")
    print("=" * 60)
    t0 = time.time()
    ckpt = train_ppo()
    train_wall = time.time() - t0
    ckpt_mb = ckpt.stat().st_size / (1024 * 1024)
    print(f"训练完成: {train_wall / 60:.1f} min, checkpoint {ckpt_mb:.1f} MB")

    print("=" * 60)
    print("Step 2/5: 录制命令序列 GIF")
    print("=" * 60)
    frames = render_command_demo()
    gif_path = PORTFOLIO_DIR / "rl_pupper_commands.gif"
    gif_utils.write_gif(
        frames,
        gif_path,
        fps=GIF_FPS,
        max_frames=144,
        width=GIF_WIDTH,
        palette_colors=12,
    )
    gif_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"GIF: {gif_path} ({gif_mb:.2f} MB)")

    print("=" * 60)
    print("Step 3/5: 录制 side-by-side comparison GIF")
    print("=" * 60)
    render_comparison_gif(ckpt)

    print("=" * 60)
    print("Step 4/5: 速度跟踪图")
    print("=" * 60)
    results = render_velocity_tracking()
    save_velocity_tracking(results)

    print("=" * 60)
    print("Step 5/5: 训练曲线")
    print("=" * 60)
    save_reward_curve()

    print("=" * 60)
    print(f"Lab 6 全部交付物已写入 {PORTFOLIO_DIR}/")
    print(f"  训练 wall-clock: {train_wall / 60:.1f} min")
    print(f"  checkpoint: {ckpt_mb:.1f} MB")
    print(f"  GIF: {gif_mb:.2f} MB")


if __name__ == "__main__":
    main()
