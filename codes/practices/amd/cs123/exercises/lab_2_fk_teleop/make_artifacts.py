"""运行 Lab 2 starter，并写出 portfolio 交付物。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.viz import gif_utils
from starter import (
    fk_validate,
    render_scripted_sweep,
    sample_workspace,
    save_workspace_plot,
)


LAB_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = LAB_DIR / "portfolio"
GIF_WRITE_KWARGS = {
    "fps": 10,
    "max_frames": 42,
    "width": 520,
}


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    max_err = fk_validate(seed=0, n=100)
    points = sample_workspace(20_000, seed=0)

    workspace_named = PORTFOLIO_DIR / "leg_workspace.png"
    gif_named = PORTFOLIO_DIR / "leg_teleop.gif"

    save_workspace_plot(points, workspace_named)
    frames = render_scripted_sweep()
    gif_utils.write_gif(frames, gif_named, **GIF_WRITE_KWARGS)

    shutil.copyfile(workspace_named, PORTFOLIO_DIR / "deliverable.png")
    shutil.copyfile(gif_named, PORTFOLIO_DIR / "deliverable.gif")

    print(f"Lab 2 交付物已写入 {PORTFOLIO_DIR}/")
    print(f"FK 校验 max_err = {max_err:.2e} m")


if __name__ == "__main__":
    main()
