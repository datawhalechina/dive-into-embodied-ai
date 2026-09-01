"""Lab 8 数值断言（不依赖 GPU / API key，跑 < 30 s）。

测试 HSV 检测、P 控制器、FSM 状态切换、物理系统集成、相机渲染。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
EXERCISES_DIR = LAB_DIR.parent

# Lab 8 目录必须在 Lab 7 之前，否则 `from starter import` 会找到 Lab 7 的 starter
sys.path.insert(0, str(LAB_DIR))
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

_LAB7_DIR = EXERCISES_DIR / "lab_7_llm_control"
if str(_LAB7_DIR) not in sys.path:
    sys.path.append(str(_LAB7_DIR))
if str(_LAB7_DIR / "tools") not in sys.path:
    sys.path.append(str(_LAB7_DIR / "tools"))

from perception.ball_detector import detect_red_ball  # noqa: E402
from perception.tracker_fsm import TrackerFSM, FSMState, visual_servo, H_REF  # noqa: E402
from starter import ChaseRobotState, _setup_head_renderer  # noqa: E402


def _make_robot() -> ChaseRobotState:
    robot = ChaseRobotState(policy_name="test")
    robot.reset()
    return robot


def test_detect_red_ball():
    """纯红色块图像 → 检测到 bbox。"""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[100:140, 140:180] = [255, 20, 20]

    box = detect_red_ball(img)
    assert box is not None, "应检测到红色块"
    assert 140 < box["cx"] < 180, f"cx 应在红色块范围内: {box['cx']}"
    assert 100 < box["cy"] < 140, f"cy 应在红色块范围内: {box['cy']}"
    assert box["area"] > _MIN_AREA_FOR_TEST, f"area 应 > 阈值: {box['area']}"
    print("  test_detect_red_ball 通过")


def test_detect_no_ball():
    """纯蓝色图像 → 返回 None。"""
    img = np.full((240, 320, 3), [30, 60, 200], dtype=np.uint8)

    box = detect_red_ball(img)
    assert box is None, f"蓝色图像不应检测到红球: {box}"
    print("  test_detect_no_ball 通过")


def test_visual_servo_center():
    """box 在画面中心 → vx/wz ≈ 0。"""
    box = {"cx": 160.0, "cy": 120.0, "w": 30, "h": H_REF, "area": 900}
    vx, wz = visual_servo(box, 320, 240)
    assert abs(wz) < 0.1, f"中心 box 的 wz 应接近 0: {wz}"
    assert abs(vx) < 0.1, f"h=H_REF 时 vx 应接近 0: {vx}"
    print("  test_visual_servo_center 通过")


def test_visual_servo_left():
    """box 在画面左侧 → wz > 0（向左转）。"""
    box = {"cx": 40.0, "cy": 120.0, "w": 20, "h": 15, "area": 300}
    vx, wz = visual_servo(box, 320, 240)
    assert wz > 0, f"左侧 box 应 wz > 0: {wz}"
    assert vx > 0, f"小 box 应 vx > 0: {vx}"
    print("  test_visual_servo_left 通过")


def test_fsm_transitions():
    """模拟 box 出现/消失序列，验证状态切换。"""
    fsm = TrackerFSM()
    W, H = 320, 240

    state, vx, wz = fsm.step(0.0, [], W, H)
    assert state == FSMState.SEARCHING, f"初始应为 SEARCHING: {state}"

    box = {"cx": 160.0, "cy": 120.0, "w": 30, "h": 25, "area": 750}
    state, vx, wz = fsm.step(0.5, [box], W, H)
    assert state == FSMState.TRACKING, f"检测到球应为 TRACKING: {state}"

    state, vx, wz = fsm.step(1.0, [], W, H)
    assert state == FSMState.TRACKING, f"短暂丢失应保持 TRACKING: {state}"

    state, vx, wz = fsm.step(3.5, [], W, H)
    assert state == FSMState.SEARCHING, f"丢失超时应切 SEARCHING: {state}"

    print("  test_fsm_transitions 通过")


def test_robot_walks():
    """ChaseRobotState 加载 pupper_chase.xml，走 1 秒不摔。"""
    robot = _make_robot()

    result = robot.step_cmd(0.3, 0.0, 1.0)
    assert result["ok"] is True, f"走 1 秒应成功: {result}"
    pose = result["final_pose"]
    assert pose[0] > 0.0, f"前进 1 秒后 x 应 > 0: {pose[0]}"
    print(f"  test_robot_walks 通过 (x={pose[0]:.3f})")


def test_camera_renders():
    """Renderer 能从 head_cam 拿到 (H, W, 3) 图像。"""
    robot = _make_robot()
    head_renderer = _setup_head_renderer(robot)
    head_cam_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")

    head_renderer.update_scene(robot.data, camera=head_cam_id)
    rgb = head_renderer.render()
    head_renderer.close()

    assert rgb.shape == (240, 320, 3), f"图像尺寸应为 (240, 320, 3): {rgb.shape}"
    assert rgb.dtype == np.uint8, f"dtype 应为 uint8: {rgb.dtype}"
    assert rgb.max() > 0, "图像不应全黑"
    print("  test_camera_renders 通过")


_MIN_AREA_FOR_TEST = 50


def main() -> None:
    print("Lab 8 数值断言:")
    test_detect_red_ball()
    test_detect_no_ball()
    test_visual_servo_center()
    test_visual_servo_left()
    test_fsm_transitions()
    test_robot_walks()
    test_camera_renders()
    print("Lab 8 检查全部通过。")


if __name__ == "__main__":
    main()
