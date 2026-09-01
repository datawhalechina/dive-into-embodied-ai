"""正运动学里反复用到的齐次变换工具和 Pupper 单腿 FK。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transform:
    """只放通用几何积木，不替具体机器人写 FK chain。"""

    @staticmethod
    def rot_x(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:3, :3] = (
            (1.0, 0.0, 0.0),
            (0.0, c, -s),
            (0.0, s, c),
        )
        return T

    @staticmethod
    def rot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:3, :3] = (
            (c, 0.0, s),
            (0.0, 1.0, 0.0),
            (-s, 0.0, c),
        )
        return T

    @staticmethod
    def rot_z(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:3, :3] = (
            (c, -s, 0.0),
            (s, c, 0.0),
            (0.0, 0.0, 1.0),
        )
        return T

    @staticmethod
    def translation(x: float, y: float, z: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = (x, y, z)
        return T

    @staticmethod
    def homogeneous_transform(
        rotation: np.ndarray | None = None,
        translation: tuple[float, float, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        T = np.eye(4)
        if rotation is not None:
            R = np.asarray(rotation, dtype=float)
            if R.shape != (3, 3):
                raise ValueError("rotation 必须是 3x3 矩阵")
            T[:3, :3] = R
        if translation is not None:
            p = np.asarray(translation, dtype=float)
            if p.shape != (3,):
                raise ValueError("translation 必须是长度为 3 的向量")
            T[:3, 3] = p
        return T

    @staticmethod
    def quat_wxyz(quat: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
        """把 MuJoCo 的 `(w, x, y, z)` 四元数转成 4x4 旋转矩阵。"""

        q = np.asarray(quat, dtype=float)
        if q.shape != (4,):
            raise ValueError("quat 必须是 (w, x, y, z)")
        norm = np.linalg.norm(q)
        if norm == 0.0:
            raise ValueError("quat 不能是零向量")
        w, x, y, z = q / norm
        R = np.array(
            (
                (1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w),
                (2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w),
                (2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y),
            ),
            dtype=float,
        )
        return Transform.homogeneous_transform(R)


# 下面这些常量逐字来自 `shared/models/leg.xml`。MJCF 导出模型把语义上的
# HAA/HFE/KFE 混合轴编码成固定 body quat + 局部 z 轴 hinge。
T_WORLD_BASE = Transform.homogeneous_transform(
    Transform.quat_wxyz((0.0, 0.0, 0.0, 1.0))[:3, :3],
    (0.0, 0.0, 0.5),
)
T_BASE_HIP_FIXED = Transform.homogeneous_transform(
    Transform.quat_wxyz((0.707105, -0.707108, 0.0, 0.0))[:3, :3],
    (0.075, 0.0835, 0.0),
)
T_HIP_UPPER_FIXED = Transform.homogeneous_transform(
    Transform.quat_wxyz((-3.89602e-06, 0.707107, -1.29867e-06, -0.707107))[:3, :3],
    (0.0, 0.0, 0.0),
)
T_UPPER_LOWER_FIXED = Transform.homogeneous_transform(
    Transform.quat_wxyz((9.38184e-07, -0.707108, 9.38187e-07, 0.707105))[:3, :3],
    (0.0, -0.0494, 0.0685),
)
T_LOWER_FOOT = Transform.translation(0.06231, 0.06216, 0.018)


def fk_leg(theta: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """返回 Pupper 单腿 foot site 在世界坐标系下的位置。"""

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,):
        raise ValueError("theta 必须是 (HAA, HFE, KFE) 三元组")
    haa, hfe, kfe = theta

    T_world_HAA = T_WORLD_BASE @ T_BASE_HIP_FIXED @ Transform.rot_z(haa)
    T_HAA_HFE = T_HIP_UPPER_FIXED @ Transform.rot_z(hfe)
    T_HFE_KFE = T_UPPER_LOWER_FIXED @ Transform.rot_z(kfe)
    T_world_foot = T_world_HAA @ T_HAA_HFE @ T_HFE_KFE @ T_LOWER_FOOT
    return T_world_foot[:3, 3].copy()
