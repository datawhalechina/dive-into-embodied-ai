"""加载 checkpoint，录命令序列 GIF + 画速度跟踪图。"""

from __future__ import annotations

import sys
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
)


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    print("录制命令序列 GIF ...")
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
    print(f"GIF 写入: {gif_path} ({gif_mb:.2f} MB)")

    print("录制 side-by-side comparison GIF（ours vs 上游 test_policy）...")
    render_comparison_gif()

    print("画速度跟踪图 ...")
    results = render_velocity_tracking()
    save_velocity_tracking(results)

    print("画训练曲线 ...")
    save_reward_curve()

    print("eval_commands 完成。")


if __name__ == "__main__":
    main()
