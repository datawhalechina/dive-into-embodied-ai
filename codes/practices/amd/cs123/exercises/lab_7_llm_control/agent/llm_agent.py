"""Lab 7 LLM agent 循环（教程 §7.3）。

run_agent(user_msg, robot, ...) 执行一轮 tool_use 循环，
返回 (final_text, messages) 供消息轨迹导出。

使用 OpenAI-compatible SDK，支持 DeepSeek / OpenAI / Ollama 等。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

load_dotenv(LAB_DIR / ".env")

from tools.robot_tools import TOOLS, RobotState, dispatch  # noqa: E402

DEFAULT_MODEL = "deepseek-chat"

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


def run_agent(
    user_msg: str,
    robot: RobotState,
    *,
    max_turns: int = 16,
    model: str | None = None,
    frame_callback=None,
) -> tuple[str, list[dict[str, Any]]]:
    """执行 agent 循环，返回 (最终文本, 完整 messages 列表)。

    frame_callback(robot) 在每个 dispatch 的物理步后被调用，用于 GIF 采帧。
    """
    from openai import OpenAI

    client = OpenAI()
    if model is None:
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    messages: list[dict[str, Any]] = [
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
            args = json.loads(tc.function.arguments)
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
# 消息轨迹格式化
# ---------------------------------------------------------------------------

def format_trace_table(messages: list[dict[str, Any]]) -> str:
    """把 messages 列表格式化成 markdown 表格。"""
    rows = ["| turn | role | type | tool_name | args | result_summary |",
            "|------|------|------|-----------|------|----------------|"]
    turn = 0
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        if role == "system":
            continue

        if role == "tool":
            rows.append(
                f"| {turn} | {role} | tool_result | — | — | {_truncate(content or '')} |"
            )
            turn += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("function", {}).get("name", "")
                        args_str = tc.get("function", {}).get("arguments", "")
                    else:
                        name = tc.function.name
                        args_str = tc.function.arguments
                    rows.append(
                        f"| {turn} | {role} | tool_call | {name} | {_truncate(args_str)} | — |"
                    )
                    turn += 1
            if content:
                rows.append(f"| {turn} | {role} | text | — | — | {_truncate(content)} |")
                turn += 1
            continue

        if isinstance(content, str) and content:
            rows.append(f"| {turn} | {role} | text | — | — | {_truncate(content)} |")
            turn += 1

    return "\n".join(rows)


def _truncate(s: str, max_len: int = 60) -> str:
    s = s.replace("\n", " ").replace("|", "\\|")
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s
