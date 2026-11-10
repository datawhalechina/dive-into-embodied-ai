"""Quaddle MJCF 模型冒烟测试。

运行：cd codes/practices/quadruped/cs123 && uv run pytest -q 7.quaddle-mjcf/tests
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

LAB_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = LAB_DIR / "quaddle.xml"


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def make_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = 0.0
    return data


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 3.0) -> None:
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def test_model_structure(model: mujoco.MjModel) -> None:
    assert model.nq == 31  # 7 free + 4 leg * 6 hinge
    assert model.nu == 4
    assert model.neq == 8  # 每腿 2 个闭链 connect
    assert model.nbody == 26  # world + body + 4 leg * 6
    assert abs(float(model.body_subtreemass[1]) - 0.1883) < 0.01


def test_keyframe_home(model: mujoco.MjModel) -> None:
    data = make_data(model)
    mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qacc))


def test_settled_stand(model: mujoco.MjModel) -> None:
    data = make_data(model)
    settle(model, data, 3.0)
    mat = data.body("body").xmat.reshape(3, 3).copy()
    roll = abs(np.degrees(np.arctan2(mat[2, 1], mat[2, 2])))
    pitch = abs(np.degrees(np.arctan2(-mat[2, 0], np.hypot(mat[2, 1], mat[2, 2]))))
    assert roll < 5.0, f"roll {roll:.1f}deg"
    assert pitch < 5.0, f"pitch {pitch:.1f}deg"
    z = float(data.body("body").xpos[2])
    assert 0.035 < z < 0.065, f"body 高度异常 {z * 1000:.0f}mm"
    # 四足足端球应接近地面（球心 - 半径 < 8mm）
    for leg in ["lf", "rf", "rb", "lb"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_col")
        foot_bottom = float(data.geom_xpos[gid][2]) - 0.0035
        assert foot_bottom < 0.008, f"{leg} 足端悬空 {foot_bottom * 1000:.1f}mm"


def test_servo_hold_under_load(model: mujoco.MjModel) -> None:
    """前腿推到 +20deg：舵机应把腿拉回目标附近（位置伺服有效）。"""
    data = make_data(model)
    settle(model, data, 2.0)
    data.ctrl[0] = math.radians(20.0)  # lf
    settle(model, data, 2.0)
    q = float(data.joint("lf_thigh").qpos[0])
    assert abs(q - math.radians(20.0)) < math.radians(6.0), f"舵机跟踪偏差 {math.degrees(q):.1f}deg"


def test_gait_does_not_diverge(model: mujoco.MjModel) -> None:
    """trot 正弦步态 5 秒：无 NaN、身体高度不发散（W1 只要求稳定性基线）。"""
    data = make_data(model)
    settle(model, data, 2.0)
    dt = model.opt.timestep
    servo = np.zeros(4)
    phases = np.array([0.0, math.pi, 0.0, math.pi])
    amp = math.radians(12.0)
    for i in range(int(5.0 / dt)):
        t = i * dt
        s = 2.0 * math.pi * 1.5 * t
        target = np.array([amp * math.sin(s + p) for p in phases])
        servo += (1.0 - math.exp(-dt / 0.03)) * (target - servo)
        data.ctrl[:] = servo
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), "步态仿真发散（NaN）"
    z = float(data.body("body").xpos[2])
    assert z > 0.01, f"身体塌陷 z={z * 1000:.0f}mm"
