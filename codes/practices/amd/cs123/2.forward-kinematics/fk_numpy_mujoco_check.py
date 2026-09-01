from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
XML_PATH = HERE / "planar_3dof.xml"


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0],
                 [s,  c, 0],
                 [0,  0, 1]]
    return T


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def fk_planar(thetas, L=(0.3, 0.25, 0.15)):
    t1, t2, t3 = thetas
    T = (
        rot_z(t1) @ trans(L[0], 0, 0)
        @ rot_z(t2) @ trans(L[1], 0, 0)
        @ rot_z(t3) @ trans(L[2], 0, 0)
    )
    return T  # 末端位置直接读 T[:3, 3]


def max_fk_error(samples=100, seed=0):
    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    end_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_site")

    max_err = 0.0
    for _ in range(samples):
        q = rng.uniform(-np.pi, np.pi, size=3)
        data.qpos[:3] = q
        mujoco.mj_forward(model, data)

        p_mj = data.site_xpos[end_id]
        p_ours = fk_planar(q)[:3, 3]
        max_err = max(max_err, np.linalg.norm(p_mj - p_ours))

    return max_err


if __name__ == "__main__":
    max_err = max_fk_error(samples=100, seed=0)
    print(f"max |p_ours - p_mj| = {max_err:.2e}")
