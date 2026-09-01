"""一键串：eval_chase → 全部交付物。"""

from __future__ import annotations

import sys
import time
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

    print("=" * 60)
    print("Lab 8 一键生成全部交付物")
    print("=" * 60)

    t0 = time.time()

    robot = ChaseRobotState(policy_name="test")
    renderer, camera = _setup_renderer(robot)
    head_renderer = _setup_head_renderer(robot)

    try:
        print("\n[1/3] 录制追球 GIF ...")
        frames, result = record_chase_gif(
            robot, renderer, camera, head_renderer,
        )
    finally:
        renderer.close()
        head_renderer.close()

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
    print(f"  GIF: {gif_path.name} ({gif_mb:.2f} MB, {len(frames)} frames)")

    timeline = result["timeline"]
    if timeline:
        print("\n[2/3] 生成图表 ...")
        plot_tracking_error(timeline, PORTFOLIO_DIR / "tracking_error.png")
        print(f"  tracking_error.png")
        plot_detection_vs_distance(timeline, PORTFOLIO_DIR / "detection_vs_distance.png")
        print(f"  detection_vs_distance.png")

    print("\n[3/3] 检查 notes.md ...")
    notes_path = PORTFOLIO_DIR / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Lab 8 反思\n\n<!-- 50–100 字：追球 demo 中最让你意外的一件事 -->\n\n",
            encoding="utf-8",
        )
        print(f"  已创建模板: {notes_path.name}")
    else:
        print(f"  已存在: {notes_path.name}")

    wall = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Lab 8 全部交付物已写入 {PORTFOLIO_DIR}/")
    print(f"  耗时: {wall:.0f} s")
    print(f"  GIF: ball_chase.gif")
    print(f"  图表: tracking_error.png, detection_vs_distance.png")
    print(f"  反思: notes.md")


if __name__ == "__main__":
    main()
