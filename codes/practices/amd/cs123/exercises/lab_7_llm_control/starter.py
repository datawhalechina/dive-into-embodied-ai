"""Lab 7 filled starter：Pupper 的自然语言简历 (LLM Control)。

教师版完整实现：加载 lab5 上游 RTNeural policy（默认 test，和 lab6 一致），
跑 5 级任务，每级录一段 GIF + 导出消息轨迹 markdown。

使用 OpenAI-compatible SDK，支持 DeepSeek / OpenAI / Ollama 等。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw

EXERCISES_DIR = Path(__file__).resolve().parents[1]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.viz import gif_utils  # noqa: E402

LAB_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = LAB_DIR / "portfolio"

load_dotenv(LAB_DIR / ".env")

sys.path.insert(0, str(LAB_DIR))
from tools.robot_tools import TOOLS, RobotState, dispatch  # noqa: E402
from agent.llm_agent import run_agent, format_trace_table, SYSTEM  # noqa: E402

GIF_FPS = 12
GIF_WIDTH = 640
GIF_HEIGHT = 480

# ---------------------------------------------------------------------------
# 5 级任务定义
# ---------------------------------------------------------------------------

TASKS = [
    {
        "level": "L1",
        "label": "L1 原子",
        "user_msg": "向前走两步然后停下",
        "description": "单工具调用",
        "gif_name": "agent_L1_stop.gif",
    },
    {
        "level": "L2",
        "label": "L2 带数字",
        "user_msg": "向前走 1 米",
        "description": "参数转换",
        "gif_name": "agent_L2_walk.gif",
    },
    {
        "level": "L3",
        "label": "L3 多步顺序",
        "user_msg": "前进 1 米，然后右转 90 度",
        "description": "多工具顺序",
        "gif_name": "agent_L3_sequence.gif",
    },
    {
        "level": "L4",
        "label": "L4 ReAct 反馈",
        "user_msg": "走到坐标 (2, 1) 附近",
        "description": "get_pose + 迭代",
        "gif_name": "agent_L4_react.gif",
    },
    {
        "level": "L5",
        "label": "L5 容错降级",
        "user_msg": "以 3 米/秒的速度飞奔向前",
        "description": "超界 → 重试",
        "gif_name": "agent_L5_retry.gif",
    },
]


# ---------------------------------------------------------------------------
# 渲染工具
# ---------------------------------------------------------------------------

def _setup_renderer(robot: RobotState) -> tuple[mujoco.Renderer, mujoco.MjvCamera]:
    """创建 offscreen renderer + camera。"""
    robot.model.vis.global_.offwidth = max(robot.model.vis.global_.offwidth, GIF_WIDTH)
    robot.model.vis.global_.offheight = max(robot.model.vis.global_.offheight, GIF_HEIGHT)
    renderer = mujoco.Renderer(robot.model, height=GIF_HEIGHT, width=GIF_WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(robot.model, camera)
    camera.distance = 1.4
    camera.elevation = -20.0
    camera.azimuth = 90.0
    return renderer, camera


def _caption_frame(
    frame: np.ndarray,
    level_label: str,
    user_msg: str,
    status: str,
    t: float,
) -> np.ndarray:
    """在帧上叠加任务等级、指令、agent 状态和时间戳。"""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = gif_utils.load_font(20)
    small = gif_utils.load_font(14)

    draw.rectangle((0, 0, image.width, 52), fill=(255, 255, 255, 220))
    draw.text((10, 4), f"[{level_label}]", fill=(200, 60, 60, 255), font=font)
    draw.text((10, 28), user_msg, fill=(18, 30, 44, 255), font=small)

    draw.rectangle((0, image.height - 32, image.width, image.height), fill=(0, 0, 0, 160))
    draw.text((10, image.height - 28), f"{status}  t={t:.1f}s", fill=(230, 230, 230, 240), font=small)
    return np.asarray(image)


# ---------------------------------------------------------------------------
# 单任务 GIF 录制
# ---------------------------------------------------------------------------

def render_task_gif(
    task: dict,
    robot: RobotState,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
) -> tuple[list[np.ndarray], str, list]:
    """运行单个任务，录制 GIF 帧，返回 (frames, final_text, messages)。"""
    robot.reset()

    frames: list[np.ndarray] = []
    t_start = robot.sim_time
    current_status = "waiting"
    next_frame_time = 0.0

    def frame_callback(cb_robot: RobotState):
        nonlocal next_frame_time, current_status
        t = cb_robot.sim_time - t_start
        if t >= next_frame_time:
            camera.lookat[:] = cb_robot.data.xpos[cb_robot.base_id]
            renderer.update_scene(cb_robot.data, camera=camera)
            raw = renderer.render()
            frames.append(_caption_frame(
                raw, task["label"], task["user_msg"], current_status, t,
            ))
            next_frame_time += 1.0 / GIF_FPS

    _capture_initial_frames(robot, renderer, camera, frames, task, t_start)

    final_text, messages = _run_agent_with_rendering(
        task["user_msg"],
        robot,
        frame_callback=frame_callback,
        status_setter=lambda s: _set_status(s),
    )

    current_status = "done"
    _capture_final_frames(robot, renderer, camera, frames, task, t_start)

    return frames, final_text, messages


def _set_status(s):
    pass


def _capture_initial_frames(robot, renderer, camera, frames, task, t_start):
    """录几帧初始静止画面。"""
    for i in range(GIF_FPS):
        t = robot.sim_time - t_start
        camera.lookat[:] = robot.data.xpos[robot.base_id]
        renderer.update_scene(robot.data, camera=camera)
        raw = renderer.render()
        frames.append(_caption_frame(raw, task["label"], task["user_msg"], "waiting", t))


def _capture_final_frames(robot, renderer, camera, frames, task, t_start):
    """录几帧结束画面。"""
    for i in range(GIF_FPS):
        t = robot.sim_time - t_start + i / GIF_FPS
        camera.lookat[:] = robot.data.xpos[robot.base_id]
        renderer.update_scene(robot.data, camera=camera)
        raw = renderer.render()
        frames.append(_caption_frame(raw, task["label"], task["user_msg"], "done", t))


def _run_agent_with_rendering(
    user_msg: str,
    robot: RobotState,
    *,
    max_turns: int = 16,
    model: str | None = None,
    frame_callback=None,
    status_setter=None,
) -> tuple[str, list]:
    """和 run_agent 相同逻辑，但 dispatch 时注入 frame_callback。"""
    from openai import OpenAI

    client = OpenAI()
    if model is None:
        model = os.getenv("OPENAI_MODEL", "deepseek-chat")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if resp.choices[0].finish_reason != "tool_calls":
            return msg.content or "", messages

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                # §7.5 软失败原则：LLM 偶尔会吐出不合法 JSON，
                # 也要把错误回报给它而不是 raise 出去。
                result = {
                    "ok": False,
                    "error": f"invalid JSON in tool arguments: {e}",
                    "raw_arguments": tc.function.arguments,
                }
            else:
                result = dispatch(
                    tc.function.name,
                    args,
                    robot,
                    frame_callback=frame_callback,
                )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "max turns reached", messages


# ---------------------------------------------------------------------------
# 消息轨迹导出
# ---------------------------------------------------------------------------

def export_traces(all_traces: dict[str, list], out_path: Path) -> None:
    """把所有任务的消息轨迹写到一个 markdown 文件。"""
    lines = ["# Agent 消息轨迹\n"]
    for task in TASKS:
        level = task["level"]
        if level not in all_traces:
            continue
        lines.append(f"## {task['label']}：{task['user_msg']}\n")
        lines.append(format_trace_table(all_traces[level]))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"消息轨迹: {out_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """快速环境检查（不需要 API key）。"""
    robot = RobotState(policy_name="test")
    robot.reset()
    print(f"Lab 7 环境检查: policy={robot.policy.path.name}")

    result = dispatch("get_pose", {}, robot)
    print(f"get_pose: {result}")
    result = dispatch("walk", {"vx": 0.3, "wz": 0.0, "duration": 1.0}, robot)
    print(f"walk(0.3, 0, 1.0): {result}")
    result = dispatch("walk", {"vx": 2.0, "wz": 0.0, "duration": 1.0}, robot)
    print(f"walk(2.0, 0, 1.0) [应报错]: {result}")


if __name__ == "__main__":
    main()
