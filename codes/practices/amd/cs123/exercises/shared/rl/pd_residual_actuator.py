"""PD 残差 actuator 配置 helper。"""

from __future__ import annotations

import mujoco
import numpy as np


def override_pd(model: mujoco.MjModel, kp: float, kv: float) -> None:
    """把 <general> actuator 的 gainprm/biasprm 改写成 (kp, kv) PD。

    对应公式: torque = kp * (ctrl - qpos) - kv * qvel
    """
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kv
