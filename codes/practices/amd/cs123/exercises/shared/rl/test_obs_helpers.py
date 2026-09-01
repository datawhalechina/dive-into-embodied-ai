"""shared/rl/obs_helpers 的数值自检。"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

EXERCISES_DIR = Path(__file__).resolve().parents[2]
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.rl.obs_helpers import base_local_gravity, foot_contact_indicator, joint_qpos_qvel_ids  # noqa: E402

MODEL_PATH = EXERCISES_DIR / "shared" / "models" / "pupper_v3_floating.xml"


def test_gravity_horizontal():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.18]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    g = base_local_gravity(model, data, base_id)
    assert abs(g[0]) < 0.1, f"gx 应接近 0: {g[0]}"
    assert abs(g[1]) < 0.1, f"gy 应接近 0: {g[1]}"
    assert g[2] <= -0.9, f"gz 应接近 -1: {g[2]}"
    return g


def test_joint_ids():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    names = ["leg_front_r_1", "leg_front_r_2", "leg_front_r_3"]
    qpos_ids, qvel_ids = joint_qpos_qvel_ids(model, names)
    assert len(qpos_ids) == 3
    assert len(qvel_ids) == 3
    assert qpos_ids[0] == 7  # floating base takes 0-6
    return qpos_ids, qvel_ids


def main():
    g = test_gravity_horizontal()
    ids = test_joint_ids()
    print("shared/rl/obs_helpers 自检通过。")
    print(f"  gravity (horizontal base) = {g}")
    print(f"  joint qpos ids (FR) = {ids[0]}, qvel ids = {ids[1]}")


if __name__ == "__main__":
    main()
