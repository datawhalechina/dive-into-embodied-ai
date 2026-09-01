import time

import mujoco
import mujoco.viewer
import numpy as np

from ik_dls_triangle import (
    fk_mj,
    ik_dls,
    interpolate_triangle,
    jac_mj,
    make_model_data,
    mocap_id,
    set_marker,
)


model, data, end_id = make_model_data()
target_mocap = mocap_id(model, "target_marker")
end_mocap = mocap_id(model, "end_marker")

vertices = np.array([
    [0.45,  0.10],
    [0.30, -0.10],
    [0.55, -0.05],
])

theta = np.zeros(3)
Kp, Kd = 60.0, 2.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        p_target = interpolate_triangle(data.time, vertices, period=3.0)
        theta, _ = ik_dls(
            lambda q: fk_mj(model, data, end_id, q),
            lambda q: jac_mj(model, data, end_id, q),
            theta,
            p_target,
            lam=0.05,
        )

        q, dq = data.qpos[:3].copy(), data.qvel[:3].copy()
        tau = Kp * (theta - q) + Kd * (-dq)
        data.ctrl[:3] = np.clip(tau, -4.0, 4.0)
        mujoco.mj_step(model, data)

        p_now = data.site_xpos[end_id, :2].copy()
        set_marker(data, target_mocap, p_target, z=0.04)
        set_marker(data, end_mocap, p_now, z=0.08)

        viewer.sync()
        wait_time = model.opt.timestep - (time.time() - step_start)
        if wait_time > 0:
            time.sleep(wait_time)
