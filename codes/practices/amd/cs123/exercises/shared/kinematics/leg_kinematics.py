"""Pupper 四条腿共用的轻量 FK / IK helper。

Lab 5 只需要把足端目标点交给 `ik_pupper_leg`。左右腿的 HAA 镜像、髋关节
偏置和关节限位都在这里处理，starter 里不再重写 IK。
"""

from __future__ import annotations

import numpy as np


LEG_ORDER = ("FL", "FR", "RL", "RR")
HIP_OFFSETS = {
    "FL": np.array((0.075, 0.0835, 0.0), dtype=float),
    "FR": np.array((0.075, -0.0835, 0.0), dtype=float),
    "RL": np.array((-0.075, 0.0725, 0.0), dtype=float),
    "RR": np.array((-0.075, -0.0725, 0.0), dtype=float),
}
THIGH_LENGTH = 0.08
CALF_LENGTH = 0.11
JOINT_LIMITS = np.array(((-0.8, 0.8), (-1.8, 1.2), (-2.6, 0.1)), dtype=float)
DEFAULT_STAND_Q = np.array((0.0, 0.18, -0.36), dtype=float)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=float)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=float)


def _check_leg(leg: str) -> str:
    name = leg.upper()
    if name not in HIP_OFFSETS:
        raise ValueError(f"leg 必须是 {LEG_ORDER} 之一，收到 {leg!r}")
    return name


def fk_pupper_leg(q: np.ndarray, *, leg: str = "FR") -> np.ndarray:
    """返回 foot site 在 base 坐标系下的位置 `(x, y, z)`。"""

    name = _check_leg(leg)
    q = np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError("q 必须是 (HAA, HFE, KFE) 三元组")
    haa, hfe, kfe = q
    p = HIP_OFFSETS[name].copy()
    rot = _rot_x(float(haa)) @ _rot_y(float(hfe))
    p = p + rot @ np.array((0.0, 0.0, -THIGH_LENGTH), dtype=float)
    rot = rot @ _rot_y(float(kfe))
    p = p + rot @ np.array((0.0, 0.0, -CALF_LENGTH), dtype=float)
    return p


def _jacobian(q: np.ndarray, *, leg: str, eps: float = 1e-5) -> np.ndarray:
    J = np.zeros((3, 3), dtype=float)
    for i in range(3):
        dq = np.zeros(3, dtype=float)
        dq[i] = eps
        J[:, i] = (fk_pupper_leg(q + dq, leg=leg) - fk_pupper_leg(q - dq, leg=leg)) / (2.0 * eps)
    return J


def _default_seed(target: np.ndarray, leg: str) -> np.ndarray:
    seed = DEFAULT_STAND_Q.copy()
    dy = float(target[1] - HIP_OFFSETS[leg][1])
    dz = float(target[2])
    seed[0] = np.clip(np.arctan2(dy, max(-dz, 1e-6)), JOINT_LIMITS[0, 0], JOINT_LIMITS[0, 1])
    return seed


def ik_pupper_leg(
    foot_xyz: np.ndarray,
    *,
    leg: str = "FR",
    q_seed: np.ndarray | None = None,
    tol: float = 1e-4,
    max_iter: int = 30,
) -> np.ndarray:
    """把 base 坐标系下的 foot 目标点反解成 `(HAA, HFE, KFE)`。

    DLS 迭代只藏在 shared helper 内。Lab 5 的学生只需要关心 `leg` 和目标足端点。
    """

    name = _check_leg(leg)
    target = np.asarray(foot_xyz, dtype=float)
    if target.shape != (3,):
        raise ValueError("foot_xyz 必须是长度为 3 的向量")
    q = _default_seed(target, name) if q_seed is None else np.asarray(q_seed, dtype=float).copy()
    q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    for _ in range(max_iter):
        err = target - fk_pupper_leg(q, leg=name)
        if float(np.linalg.norm(err)) < tol:
            break
        J = _jacobian(q, leg=name)
        damping = 2e-3
        lhs = J @ J.T + (damping**2) * np.eye(3)
        try:
            step = J.T @ np.linalg.solve(lhs, err)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(J, rcond=1e-8) @ err
        q = np.clip(q + 0.75 * step, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return q


__all__ = [
    "CALF_LENGTH",
    "DEFAULT_STAND_Q",
    "HIP_OFFSETS",
    "JOINT_LIMITS",
    "LEG_ORDER",
    "THIGH_LENGTH",
    "fk_pupper_leg",
    "ik_pupper_leg",
]
