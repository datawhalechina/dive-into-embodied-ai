"""运行 Lab 3 starter，并写出 portfolio 交付物。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.viz import gif_utils  # noqa: E402
from starter import PORTFOLIO_DIR, save_raibert_vs_triangle_plot, simulate_stepping  # noqa: E402

GIF_WRITE_KWARGS = {
    "fps": 10,
    "max_frames": 54,
    "width": 480,
}
SWEEP_GIF_WRITE_KWARGS = {
    "fps": 10,
    "max_frames": 72,
    "width": 480,
}


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    normal = simulate_stepping(capture_frames=True, broken_tail=True)
    sweep = simulate_stepping(seconds=12.0, capture_frames=True, sweep_step_length=True)

    gif_named = PORTFOLIO_DIR / "leg_stepping.gif"
    sweep_named = PORTFOLIO_DIR / "leg_stepping_sweep.gif"
    plot_named = PORTFOLIO_DIR / "raibert_vs_triangle.png"

    if normal.frames is None or sweep.frames is None:
        raise RuntimeError("渲染没有返回 GIF 帧")
    gif_utils.write_gif(normal.frames, gif_named, **GIF_WRITE_KWARGS)
    gif_utils.write_gif(sweep.frames, sweep_named, **SWEEP_GIF_WRITE_KWARGS)
    save_raibert_vs_triangle_plot(plot_named)

    shutil.copyfile(gif_named, PORTFOLIO_DIR / "deliverable.gif")
    shutil.copyfile(plot_named, PORTFOLIO_DIR / "deliverable.png")

    max_residual = float(normal.ik_residual.max())
    print(f"Lab 3 交付物已写入 {PORTFOLIO_DIR}/")
    print(f"IK 残差 max = {max_residual:.2e} m")
    print(f"leg_stepping.gif 帧数 = {len(normal.frames)}")
    print(f"leg_stepping_sweep.gif 帧数 = {len(sweep.frames)}")


if __name__ == "__main__":
    main()
