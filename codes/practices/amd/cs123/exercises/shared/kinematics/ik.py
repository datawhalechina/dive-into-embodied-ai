"""IK 共用几何工具。

Lab 3 的 DLS 迭代本体留在 `starter.py` 里做 TODO；这里不藏答案。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

IkStepFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]]


def damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """返回 DLS 形式的阻尼伪逆 `J.T @ inv(J @ J.T + lam^2 I)`。"""

    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError("J 必须是二维矩阵")
    if lam < 0.0:
        raise ValueError("lam 必须非负")
    JJt = J @ J.T
    return J.T @ np.linalg.solve(JJt + (lam**2) * np.eye(JJt.shape[0]), np.eye(JJt.shape[0]))


def pose_error(target_xyz: np.ndarray, current_xyz: np.ndarray) -> np.ndarray:
    """返回世界系下的 3D 位置误差 `target - current`。"""

    target = np.asarray(target_xyz, dtype=float)
    current = np.asarray(current_xyz, dtype=float)
    if target.shape != (3,) or current.shape != (3,):
        raise ValueError("target_xyz 和 current_xyz 都必须是长度为 3 的向量")
    return target - current


def dls_ik(
    q0: np.ndarray,
    target_xyz: np.ndarray,
    step_fn: IkStepFn,
    *,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> tuple[np.ndarray, int, float, bool]:
    """通用 IK 迭代框架，具体 DLS 一步由 Lab starter 传入。"""

    q = np.asarray(q0, dtype=float).copy()
    target = np.asarray(target_xyz, dtype=float)
    residual = float("inf")
    for k in range(max_iter):
        q, residual = step_fn(q, target)
        if residual < tol:
            return q, k + 1, residual, True
    return q, max_iter, residual, False
