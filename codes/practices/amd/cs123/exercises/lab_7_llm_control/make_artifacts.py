"""一键串：eval_tasks → 全部交付物。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from shared.viz import gif_utils  # noqa: E402
from tools.robot_tools import RobotState  # noqa: E402
from starter import (  # noqa: E402
    GIF_FPS,
    GIF_WIDTH,
    PORTFOLIO_DIR,
    TASKS,
    render_task_gif,
    export_traces,
    _setup_renderer,
)


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    robot = RobotState(policy_name="test")
    renderer, camera = _setup_renderer(robot)

    print("=" * 60)
    print("Lab 7 一键生成全部交付物")
    print("=" * 60)

    all_traces: dict[str, list] = {}
    t0 = time.time()

    try:
        for i, task in enumerate(TASKS):
            print(f"\n[{i + 1}/{len(TASKS)}] {task['label']}: {task['user_msg']}")

            frames, final_text, messages = render_task_gif(
                task, robot, renderer, camera,
            )
            all_traces[task["level"]] = messages

            gif_path = PORTFOLIO_DIR / task["gif_name"]
            gif_utils.write_gif(
                frames,
                gif_path,
                fps=GIF_FPS,
                max_frames=120,
                width=GIF_WIDTH,
                palette_colors=12,
            )
            gif_mb = gif_path.stat().st_size / 1024 / 1024
            print(f"  GIF: {gif_path.name} ({gif_mb:.2f} MB)")
            print(f"  回复: {final_text}")
    finally:
        renderer.close()

    traces_path = PORTFOLIO_DIR / "agent_traces.md"
    export_traces(all_traces, traces_path)

    wall = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Lab 7 全部交付物已写入 {PORTFOLIO_DIR}/")
    print(f"  耗时: {wall:.0f} s")
    print(f"  GIF: {len(TASKS)} 个")
    print(f"  消息轨迹: {traces_path.name}")


if __name__ == "__main__":
    main()
