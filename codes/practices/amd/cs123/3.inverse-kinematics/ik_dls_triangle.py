from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
XML_PATH = HERE / "planar_3dof_ik.xml"


def make_model_data():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    end_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_site")
    return model, data, end_id


def fk_mj(model, data, end_id, theta):
    data.qpos[:3] = theta
    data.qvel[:3] = 0.0
    mujoco.mj_forward(model, data)
    return data.site_xpos[end_id, :2].copy()


def jac_mj(model, data, end_id, theta):
    data.qpos[:3] = theta
    data.qvel[:3] = 0.0
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, end_id)
    return jacp[:2, :3].copy()


def ik_dls(fk_fn, jac_fn, theta0, p_target, lam=0.05, step=0.3, tol=1e-4, max_iter=200):
    theta = theta0.copy()
    for k in range(max_iter):
        p = fk_fn(theta)
        e = p_target - p
        if np.linalg.norm(e) < tol:
            return theta, k

        J = jac_fn(theta)
        JJt = J @ J.T
        dtheta = J.T @ np.linalg.solve(
            JJt + (lam ** 2) * np.eye(JJt.shape[0]),
            e,
        )
        theta = theta + step * dtheta

    return theta, max_iter


def interpolate_triangle(t, vertices, period=3.0):
    """3 个顶点循环,每条边走 period/3 秒。返回当前目标点。"""
    seg = (t % period) / (period / 3)
    i = int(seg) % 3
    s = seg - int(seg)
    p0, p1 = vertices[i], vertices[(i + 1) % 3]
    return (1 - s) * p0 + s * p1


def mocap_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return model.body_mocapid[body_id]


def set_marker(data, mocap_id_, xy, z):
    data.mocap_pos[mocap_id_] = [xy[0], xy[1], z]


def run_tracking(duration=6.0):
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
    log = []

    while data.time < duration:
        p_target = interpolate_triangle(data.time, vertices, period=3.0)
        theta, iters = ik_dls(
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
        log.append((data.time, p_target.copy(), p_now.copy(), iters))

    return log


if __name__ == "__main__":
    log = run_tracking(duration=6.0)
    err = [np.linalg.norm(target - actual) for _, target, actual, _ in log]
    print(f"final error = {err[-1]:.4f} m")
    print(f"mean error  = {np.mean(err):.4f} m")
