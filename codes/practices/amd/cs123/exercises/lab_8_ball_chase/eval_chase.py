"""跑追球 demo，录 GIF + 导出图表。"""

from __future__ import annotations

import sys
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

_LAB7_DIR = EXERCISES_DIR / "lab_7_llm_control"
if str(_LAB7_DIR) not in sys.path:
    sys.path.append(str(_LAB7_DIR))
if str(_LAB7_DIR / "tools") not in sys.path:
    sys.path.append(str(_LAB7_DIR / "tools"))

from shared.viz import gif_utils  # noqa: E402
from starter import (  # noqa: E402
    ChaseRobotState,
    GIF_FPS,
    GIF_WIDTH,
    PORTFOLIO_DIR,
    record_chase_gif,
    plot_tracking_error,
    plot_detection_vs_distance,
    _setup_renderer,
    _setup_head_renderer,
)


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    robot = ChaseRobotState(policy_name="test")
    renderer, camera = _setup_renderer(robot)
    head_renderer = _setup_head_renderer(robot)

    print("=" * 60)
    print("Lab 8 追球 Demo")
    print("=" * 60)

    try:
        frames, result = record_chase_gif(
            robot, renderer, camera, head_renderer,
        )
    finally:
        renderer.close()
        head_renderer.close()

    print(f"  ok={result['ok']}, timeline 长度={len(result['timeline'])}")
    print(f"  录制帧数={len(frames)}")

    gif_path = PORTFOLIO_DIR / "ball_chase.gif"
    gif_utils.write_gif(
        frames,
        gif_path,
        fps=GIF_FPS,
        max_frames=100,
        width=GIF_WIDTH,
        palette_colors=64,
    )
    gif_mb = gif_path.stat().st_size / 1024 / 1024
    print(f"  GIF: {gif_path} ({gif_mb:.2f} MB)")

    timeline = result["timeline"]
    if timeline:
        plot_tracking_error(timeline, PORTFOLIO_DIR / "tracking_error.png")
        print(f"  图表: tracking_error.png")
        plot_detection_vs_distance(timeline, PORTFOLIO_DIR / "detection_vs_distance.png")
        print(f"  图表: detection_vs_distance.png")

    print(f"\n{'=' * 60}")
    print(f"交付物在 {PORTFOLIO_DIR}/")


if __name__ == "__main__":
    main()
