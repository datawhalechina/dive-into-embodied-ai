"""Longitudinal plane perturbations; positive slope rises along robot forward (-Y)."""
import math

import mujoco
import numpy as np


def configure_ground(model, data, slope_deg=0.0):
    """Rotate the plane and initial floating base together, before stepping.

    The returned rotation maps ground-frame vectors into world coordinates.
    This is a test surface, not a measured physical ramp or actuator model.
    """
    if not math.isfinite(slope_deg) or abs(slope_deg) >= 90:
        raise ValueError("slope_deg must be finite and between -90 and 90 degrees")
    angle = -math.radians(slope_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    quat = np.array([math.cos(angle / 2), math.sin(angle / 2), 0., 0.])
    # These models have a world-attached plane and one leading free joint.
    ground = model.geom('ground').id
    model.geom_quat[ground] = quat
    # Recompute compiled frame shortcuts after changing a world-attached geom.
    # Otherwise an initially body-aligned plane can silently stay horizontal.
    mujoco.mj_setConst(model, data)
    data.qpos[:3] = rotation @ data.qpos[:3]
    root_quat = np.empty(4)
    mujoco.mju_mulQuat(root_quat, quat, data.qpos[3:7].copy())
    data.qpos[3:7] = root_quat
    mujoco.mj_forward(model, data)
    return rotation
