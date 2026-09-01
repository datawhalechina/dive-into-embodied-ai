"""运行 Lab 1 starter，并写出 portfolio 交付物。"""

from __future__ import annotations

import shutil
from pathlib import Path

from starter import GIF_FREQUENCY_HZ, run_experiment, save_bode_plot, save_gif


LAB_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = LAB_DIR / "portfolio"
GIF_WRITE_KWARGS = {
    "fps": 8,
    "max_frames": 45,
    "width": 560,
}


def _freq_slug(freq: float) -> str:
    return f"{freq:g}".replace(".", "p")


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    data = run_experiment(render=True)
    gains = data["gains"]
    inertia = data["inertia"]
    bode_points = data["bode_points"]
    sine_gif = data["sine_gif"]

    bode_named = PORTFOLIO_DIR / "bode_pupper_hfe.png"
    gif_named = PORTFOLIO_DIR / f"hfe_sine_{_freq_slug(GIF_FREQUENCY_HZ)}hz.gif"

    save_bode_plot(bode_points, bode_named, gains, inertia)
    save_gif(sine_gif, gif_named, **GIF_WRITE_KWARGS)

    shutil.copyfile(bode_named, PORTFOLIO_DIR / "deliverable.png")
    shutil.copyfile(gif_named, PORTFOLIO_DIR / "deliverable.gif")

    print(f"Lab 1 交付物已写入 {PORTFOLIO_DIR}/")


if __name__ == "__main__":
    main()
