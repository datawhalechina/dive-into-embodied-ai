"""Lab 7 TODO starter：Pupper 的自然语言简历 (LLM Control)。

学生只需要补三处 TODO：
  TODO 1: dispatch()——工具参数校验 + 执行
  TODO 2: run_agent()——OpenAI SDK tool_use 循环
  TODO 3: render_task_gif()——单任务 GIF 录制

物理后端复用 lab5 的 RTNeural policy（默认 test，和 lab6 一致），
跑 50 Hz policy / 500 Hz physics 的同一套循环。
其余 GIF 渲染管线、消息轨迹导出、任务定义都已经接好。

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
from tools.robot_tools import TOOLS, RobotState  # noqa: E402

DEFAULT_MODEL = "deepseek-chat"

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

VX_RANGE = (-1.5, 1.5)
WZ_RANGE = (-2.0, 2.0)
DURATION_RANGE = (0.0, 10.0)

SYSTEM = (
    "You are a controller for a small quadruped robot in a MuJoCo simulation. "
    "Translate the user's natural-language commands into one or more tool calls. "
    "Use SI units. After each tool result, decide whether the goal is reached; "
    "if not, call another tool. "
    "If a tool returns an error (ok=false), automatically retry with corrected "
    "parameters (e.g. clamp to the valid range). Never ask the user for permission. "
    "For navigation tasks, use get_pose to check progress and call walk repeatedly "
    "until you are close enough. When done, reply in one short sentence."
)


# ---------------------------------------------------------------------------
# TODO 1: dispatch
# ---------------------------------------------------------------------------

def dispatch(name: str, args: dict, robot: RobotState, *, frame_callback=None) -> dict:
    """执行工具调用，返回 JSON-able 结果。绝不 raise，只在返回值里写错误。

    提示（教程 §7.5）：
    - walk: 校验 vx ∈ [-1.5, 1.5]、wz ∈ [-2.0, 2.0]、duration ∈ (0, 10]
      越界 → {"ok": False, "error": "vx out of range [-1.5, 1.5]"}
      通过 → 调用 robot.step_cmd(vx, wz, duration, frame_callback=frame_callback)
    - stop: robot.step_cmd(0.0, 0.0, 0.5, ...)
    - get_pose: 返回 {"ok": True, "pose": robot.get_pose()}
    - wait: 校验 seconds ∈ (0, 10]，robot.step_cmd(0.0, 0.0, seconds, ...)
    - 未知工具 → {"ok": False, "error": "unknown tool: {name}"}
    - 所有异常用 try/except 包住，返回 {"ok": False, "error": str(e)}

    想一想：为什么 dispatch 绝不 raise？
    如果 raise 了，LLM 收不到错误信息，就没法自动重试。
    """
    raise NotImplementedError(
        "TODO 1: 实现 dispatch。\n"
        "需要处理 4 个工具（walk / stop / get_pose / wait），\n"
        "对 walk 做参数校验（vx / wz / duration 量程），\n"
        "越界返回 {'ok': False, 'error': '...'}，通过则调用 robot.step_cmd(vx, wz, duration)。\n"
        "提示：get_pose 用 robot.get_pose()，stop 用 robot.step_cmd(0, 0, 0.5)。"
    )


# ---------------------------------------------------------------------------
# TODO 2: run_agent
# ---------------------------------------------------------------------------

def run_agent(
    user_msg: str,
    robot: RobotState,
    *,
    max_turns: int = 16,
    model: str | None = None,
    frame_callback=None,
) -> tuple[str, list]:
    """执行 agent 循环，返回 (最终文本, 完整 messages 列表)。

    提示（教程 §7.3）：
    - 用 openai.OpenAI() 创建客户端
    - messages 初始化为 [{"role": "system", ...}, {"role": "user", ...}]
    - 循环最多 max_turns 轮：
      1. client.chat.completions.create(model=model, tools=TOOLS, messages=messages)
      2. 把 resp.choices[0].message 追加到 messages（用 msg.model_dump()）
      3. 如果 finish_reason != "tool_calls"，返回 msg.content
      4. 否则遍历 msg.tool_calls，对每个调用 dispatch
      5. 把 tool results 追加到 messages（role="tool"）
    - 超过 max_turns 返回 "max turns reached"

    想一想：为什么 dispatch 绝不 raise？
    如果 raise 了，LLM 收不到错误信息，就没法自动重试。
    """
    raise NotImplementedError(
        "TODO 2: 实现 run_agent。\n"
        "用 OpenAI SDK 的 tool_use 循环：\n"
        "  发消息 → 检查 finish_reason → 执行 tool_calls → 组装 tool result → 喂回去。\n"
        "提示：finish_reason == 'tool_calls' 时遍历 msg.tool_calls，\n"
        "对每个 tc 调用 dispatch(tc.function.name, json.loads(tc.function.arguments), robot)，\n"
        "把结果包成 {'role': 'tool', 'tool_call_id': tc.id, 'content': json.dumps(result)}。"
    )


# ---------------------------------------------------------------------------
# TODO 3: render_task_gif
# ---------------------------------------------------------------------------

def render_task_gif(
    task: dict,
    robot: RobotState,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
) -> tuple[list[np.ndarray], str, list]:
    """运行单个任务，录制 GIF 帧，返回 (frames, final_text, messages)。

    提示：
    1. robot.reset() 重置环境
    2. 录几帧初始静止画面（约 1 秒 = GIF_FPS 帧）
    3. 定义 frame_callback(cb_robot)：
       - 计算 t = cb_robot.sim_time - t_start
       - 如果 t >= next_frame_time：渲染一帧、加字幕、追加到 frames
       - next_frame_time += 1.0 / GIF_FPS
    4. 调用 run_agent(task["user_msg"], robot, frame_callback=frame_callback)
    5. 录几帧结束画面
    6. 返回 (frames, final_text, messages)

    字幕用 _caption_frame(frame, level_label, user_msg, status, t)。
    渲染用 renderer.update_scene(robot.data, camera=camera) + renderer.render()。
    camera.lookat[:] = robot.data.xpos[robot.base_id] 让相机跟随机器人。
    """
    raise NotImplementedError(
        "TODO 3: 实现 render_task_gif。\n"
        "reset robot → 录初始帧 → 定义 frame_callback → 调用 run_agent → 录结束帧。\n"
        "提示：frame_callback 在每个物理步后被调用，按 GIF_FPS 采帧。\n"
        "用 renderer.update_scene + renderer.render 做 offscreen 渲染，\n"
        "用 _caption_frame 加字幕（level_label / user_msg / status / t）。"
    )


# ---------------------------------------------------------------------------
# 渲染工具（已实现，供 TODO 3 使用）
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
# 消息轨迹导出
# ---------------------------------------------------------------------------

def format_trace_table(messages: list) -> str:
    """把 messages 列表格式化成 markdown 表格。"""
    rows = ["| turn | role | type | tool_name | args | result_summary |",
            "|------|------|------|-----------|------|----------------|"]
    turn = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            rows.append(f"| {turn} | {role} | text | — | — | {_truncate(content)} |")
            turn += 1
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        result_str = block.get("content", "")
                        rows.append(f"| {turn} | {role} | tool_result | — | — | {_truncate(result_str)} |")
                        turn += 1
                else:
                    btype = getattr(block, "type", "")
                    if btype == "tool_use":
                        args_str = json.dumps(block.input, ensure_ascii=False)
                        rows.append(f"| {turn} | {role} | tool_use | {block.name} | {_truncate(args_str)} | — |")
                        turn += 1
                    elif btype == "text":
                        rows.append(f"| {turn} | {role} | text | — | — | {_truncate(block.text)} |")
                        turn += 1
    return "\n".join(rows)


def _truncate(s: str, max_len: int = 60) -> str:
    s = s.replace("\n", " ").replace("|", "\\|")
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


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


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """快速环境检查（不需要 API key）。"""
    robot = RobotState(policy_name="test")
    robot.reset()
    print(f"Lab 7 环境检查: policy={robot.policy.path.name}")
    print("dispatch / run_agent 需要你补完 TODO 1–3 后才能运行。")


if __name__ == "__main__":
    main()
