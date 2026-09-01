"""Lab 7 数值断言（不依赖 API key，跑 < 30 s）。

测试 dispatch 逻辑（参数校验、软失败）、工具 schema 完整性、RobotState 集成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from tools.robot_tools import TOOLS, VX_RANGE, WZ_RANGE, DURATION_RANGE, RobotState, dispatch  # noqa: E402


def _make_robot() -> RobotState:
    robot = RobotState(policy_name="test")
    robot.reset()
    return robot


def test_tool_schemas():
    """TOOLS 列表完整性：4 个工具，每个有 type=function + function.name/description/parameters。"""
    assert len(TOOLS) == 4, f"TOOLS 应有 4 个工具，实际 {len(TOOLS)}"
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"walk", "stop", "get_pose", "wait"}, f"工具名不对: {names}"
    for tool in TOOLS:
        assert tool.get("type") == "function", f"工具应有 type=function: {tool}"
        fn = tool["function"]
        assert "description" in fn, f"{fn['name']} 缺 description"
        assert "parameters" in fn, f"{fn['name']} 缺 parameters"
    walk_fn = next(t["function"] for t in TOOLS if t["function"]["name"] == "walk")
    walk_props = walk_fn["parameters"]["properties"]
    assert "vx" in walk_props, "walk 缺 vx 参数"
    assert "wz" in walk_props, "walk 缺 wz 参数"
    assert "duration" in walk_props, "walk 缺 duration 参数"
    assert walk_fn["parameters"]["required"] == ["vx", "wz", "duration"]
    print("  test_tool_schemas 通过")


def test_dispatch_validation():
    """dispatch 参数校验：越界返回 ok=False，不 raise。"""
    robot = _make_robot()

    r = dispatch("walk", {"vx": 2.0, "wz": 0.0, "duration": 1.0}, robot)
    assert r["ok"] is False, f"vx=2.0 应返回 ok=False: {r}"
    assert "vx" in r["error"].lower() and "range" in r["error"].lower(), f"错误信息应提到 vx range: {r['error']}"

    r = dispatch("walk", {"vx": 0.0, "wz": 3.0, "duration": 1.0}, robot)
    assert r["ok"] is False, f"wz=3.0 应返回 ok=False: {r}"

    r = dispatch("walk", {"vx": 0.0, "wz": 0.0, "duration": 0.0}, robot)
    assert r["ok"] is False, f"duration=0 应返回 ok=False: {r}"

    r = dispatch("walk", {"vx": 0.0, "wz": 0.0, "duration": 15.0}, robot)
    assert r["ok"] is False, f"duration=15 应返回 ok=False: {r}"

    r = dispatch("unknown_tool", {}, robot)
    assert r["ok"] is False, f"未知工具应返回 ok=False: {r}"
    assert "unknown" in r["error"].lower(), f"错误信息应提到 unknown: {r['error']}"

    print("  test_dispatch_validation 通过")


def test_dispatch_get_pose():
    """get_pose 返回合理的 [x, y, yaw]。"""
    robot = _make_robot()

    r = dispatch("get_pose", {}, robot)
    assert r["ok"] is True, f"get_pose 应返回 ok=True: {r}"
    assert "pose" in r, f"get_pose 应返回 pose: {r}"
    pose = r["pose"]
    assert len(pose) == 3, f"pose 应有 3 个元素: {pose}"
    assert all(np.isfinite(v) for v in pose), f"pose 含 inf/nan: {pose}"
    assert abs(pose[0]) < 1.0, f"reset 后 x 应接近 0: {pose[0]}"
    assert abs(pose[1]) < 1.0, f"reset 后 y 应接近 0: {pose[1]}"
    print("  test_dispatch_get_pose 通过")


def test_dispatch_walk():
    """walk 正常参数能执行，返回 final_pose。"""
    robot = _make_robot()

    r = dispatch("walk", {"vx": 0.3, "wz": 0.0, "duration": 3.0}, robot)
    assert r["ok"] is True, f"walk(0.3, 0, 3.0) 应返回 ok=True: {r}"
    assert "final_pose" in r, f"walk 应返回 final_pose: {r}"
    pose = r["final_pose"]
    assert len(pose) == 3, f"final_pose 应有 3 个元素: {pose}"
    assert all(np.isfinite(v) for v in pose), f"final_pose 含 inf/nan: {pose}"
    assert pose[0] > 0.0, f"前进 3 秒后 x 应 > 0: {pose[0]}"
    print(f"  test_dispatch_walk 通过 (final x={pose[0]:.3f})")


def test_dispatch_stop():
    """stop 返回 ok=True。"""
    robot = _make_robot()

    r = dispatch("stop", {}, robot)
    assert r["ok"] is True, f"stop 应返回 ok=True: {r}"
    print("  test_dispatch_stop 通过")


def test_dispatch_wait():
    """wait 正常参数能执行。"""
    robot = _make_robot()

    r = dispatch("wait", {"seconds": 0.5}, robot)
    assert r["ok"] is True, f"wait(0.5) 应返回 ok=True: {r}"

    r = dispatch("wait", {"seconds": 0.0}, robot)
    assert r["ok"] is False, f"wait(0) 应返回 ok=False: {r}"
    print("  test_dispatch_wait 通过")


def test_soft_failure_no_raise():
    """dispatch 绝不 raise，即使传入奇怪的参数。"""
    robot = _make_robot()

    r = dispatch("walk", {"vx": "not_a_number", "wz": 0.0, "duration": 1.0}, robot)
    assert isinstance(r, dict), f"dispatch 应返回 dict: {type(r)}"
    assert "ok" in r, f"dispatch 返回值应有 ok 字段: {r}"
    print("  test_soft_failure_no_raise 通过")


def main() -> None:
    print("Lab 7 数值断言:")
    test_tool_schemas()
    test_dispatch_validation()
    test_dispatch_get_pose()
    test_dispatch_walk()
    test_dispatch_stop()
    test_dispatch_wait()
    test_soft_failure_no_raise()
    print("Lab 7 检查全部通过。")


if __name__ == "__main__":
    main()
