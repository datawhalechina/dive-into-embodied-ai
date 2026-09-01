"""RL 观测向量的轻量 helper。"""

from __future__ import annotations

import mujoco
import numpy as np


def base_local_gravity(model: mujoco.MjModel, data: mujoco.MjData, base_body: str | int) -> np.ndarray:
    """返回重力方向在 base 坐标系下的投影 (3,)。

    水平站立时约为 (0, 0, -9.81)。
    """
    if isinstance(base_body, str):
        base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    R = data.xmat[base_body].reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return R.T @ gravity_world


def foot_contact_indicator(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_bodies: list[str | int],
    threshold: float = 0.5,
) -> np.ndarray:
    """返回 (N,) 布尔数组，True 表示该足端有接触力。"""
    result = np.zeros(len(foot_bodies), dtype=np.float32)
    for i, body in enumerate(foot_bodies):
        if isinstance(body, str):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if data.cfrc_ext[body, 2] > threshold:
            result[i] = 1.0
    return result


def joint_qpos_qvel_ids(
    model: mujoco.MjModel,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """返回一组关节的 qpos 和 qvel 索引数组。"""
    qpos_ids = []
    qvel_ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"关节 {name!r} 不存在")
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        qvel_ids.append(int(model.jnt_dofadr[jid]))
    return np.array(qpos_ids, dtype=int), np.array(qvel_ids, dtype=int)
