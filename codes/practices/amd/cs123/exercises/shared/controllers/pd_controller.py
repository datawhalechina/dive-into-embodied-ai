"""Lab 共用的控制脚手架。

PD 公式本身刻意留在每个 Lab 的 starter TODO 里。这个模块只放可复用的部分：
类型化配置、目标信号工具函数和少量数值工具，不把学生需要亲手写的控制律藏起来。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PDGains:
    """单关节的位置增益和速度增益。"""

    kp: float
    kd: float


@dataclass(frozen=True)
class JointTarget:
    """某一时刻的期望关节位置和速度。"""

    q: float
    qdot: float


@dataclass(frozen=True)
class SimulationConfig:
    """单关节仿真复用的时间步长和执行器限制。"""

    dt: float = 0.005
    max_torque: float = 3.0
    joint_name: str = "HFE"
    actuator_name: str = "hfe_motor"


@dataclass
class TrackingTrace:
    """一次跟踪仿真返回的时间序列。"""

    time: np.ndarray
    q_des: np.ndarray
    qdot_des: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    torque: np.ndarray
    kp: float
    kd: float
    inertia: float
    frequency_hz: float | None = None
    frames: list[np.ndarray] | None = None

    @property
    def error(self) -> np.ndarray:
        return self.q_des - self.q


@dataclass(frozen=True)
class BodePoint:
    """一个稳态正弦跟踪频点的测量结果。"""

    frequency_hz: float
    gain: float
    gain_db: float
    phase_deg: float


TargetFn = Callable[[float], JointTarget]


def constant_target(q_des: float) -> TargetFn:
    """返回常值位置跟踪的目标函数。"""

    def target(_t: float) -> JointTarget:
        return JointTarget(q=q_des, qdot=0.0)

    return target


def sine_target(amplitude: float, frequency_hz: float) -> TargetFn:
    """返回 q=A sin(2 pi f t) 及其导数 qdot。"""

    omega = 2.0 * np.pi * frequency_hz

    def target(t: float) -> JointTarget:
        return JointTarget(
            q=float(amplitude * np.sin(omega * t)),
            qdot=float(amplitude * omega * np.cos(omega * t)),
        )

    return target


def ensure_parent(path: Path) -> None:
    """创建输出文件的父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)


def steady_state_window(time: np.ndarray, settle_seconds: float) -> np.ndarray:
    """选出初始收敛时间之后的样本。"""

    if len(time) == 0:
        return np.array([], dtype=bool)
    return time >= float(settle_seconds)


def estimate_sine_response(
    time: np.ndarray,
    target: np.ndarray,
    actual: np.ndarray,
    frequency_hz: float,
    settle_seconds: float,
) -> BodePoint:
    """拟合 sin/cos 系数，返回增益和相位差。

    拟合只使用稳态尾段。这里不用峰值拾取，所以即使采样频率和日志间隔没有整齐对齐，
    估计也比较稳定。
    """

    mask = steady_state_window(time, settle_seconds)
    t = np.asarray(time[mask], dtype=float)
    y_ref = np.asarray(target[mask], dtype=float)
    y_out = np.asarray(actual[mask], dtype=float)
    if len(t) < 4:
        raise ValueError("稳态样本太少，无法拟合正弦响应")

    omega = 2.0 * np.pi * frequency_hz
    basis = np.column_stack(
        [
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones_like(t),
        ]
    )
    ref_coef, *_ = np.linalg.lstsq(basis, y_ref, rcond=None)
    out_coef, *_ = np.linalg.lstsq(basis, y_out, rcond=None)

    ref_amp = float(np.hypot(ref_coef[0], ref_coef[1]))
    out_amp = float(np.hypot(out_coef[0], out_coef[1]))
    if ref_amp <= 1e-12:
        raise ValueError("参考正弦幅值太小")

    ref_phase = float(np.arctan2(ref_coef[1], ref_coef[0]))
    out_phase = float(np.arctan2(out_coef[1], out_coef[0]))
    phase = np.rad2deg(out_phase - ref_phase)
    phase = float((phase + 180.0) % 360.0 - 180.0)

    gain = out_amp / ref_amp
    gain_db = 20.0 * float(np.log10(max(gain, 1e-12)))
    return BodePoint(
        frequency_hz=float(frequency_hz),
        gain=float(gain),
        gain_db=gain_db,
        phase_deg=phase,
    )
