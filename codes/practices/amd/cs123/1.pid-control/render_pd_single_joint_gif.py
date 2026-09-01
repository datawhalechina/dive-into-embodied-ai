from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
OUT = HERE / "pd_single_joint.gif"

model = mujoco.MjModel.from_xml_path(str(HERE / "pendulum.xml"))
data = mujoco.MjData(model)

q_des = 0.8
Kp, Kd = 20.0, 1.0

pos_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_act")
model.actuator_gainprm[pos_act_id, :] = 0.0
model.actuator_biasprm[pos_act_id, :] = 0.0

camera = mujoco.MjvCamera()
camera.lookat[:] = [0.0, 0.0, 0.75]
camera.distance = 1.8
camera.azimuth = 90
camera.elevation = -20

frames = []
renderer = mujoco.Renderer(model, height=480, width=640)

while data.time < 4.0:
    q = data.qpos[0]
    dq = data.qvel[0]
    tau = Kp * (q_des - q) + Kd * (0.0 - dq)
    data.ctrl[1] = np.clip(tau, -5.0, 5.0)

    mujoco.mj_step(model, data)

    if len(frames) < data.time * 30:
        renderer.update_scene(data, camera=camera)
        frames.append(Image.fromarray(renderer.render()))

frames[0].save(
    OUT,
    save_all=True,
    append_images=frames[1:],
    duration=1000 // 30,
    loop=0,
)

print(f"saved {OUT} ({len(frames)} frames)")
