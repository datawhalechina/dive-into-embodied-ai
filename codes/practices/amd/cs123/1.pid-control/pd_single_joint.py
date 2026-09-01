from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("pendulum.xml")))
data = mujoco.MjData(model)

q_des = 0.8  # 目标角度(rad),约 45.8°
Kp, Kd = 20.0, 1.0

use_motor = True  # True: 用 motor 手搓 PD; False: 用 position 执行器

if use_motor:
    # 不关掉 position 执行器的话,它会和手写 motor PD 同时出力.
    pos_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_act")
    model.actuator_gainprm[pos_act_id, :] = 0.0
    model.actuator_biasprm[pos_act_id, :] = 0.0

log = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 8.0:
        step_start = time.time()

        q = data.qpos[0]
        dq = data.qvel[0]

        if use_motor:
            tau = Kp * (q_des - q) + Kd * (0.0 - dq)
            data.ctrl[1] = np.clip(tau, -5.0, 5.0)  # motor 通道
        else:
            data.ctrl[0] = q_des  # position 通道

        mujoco.mj_step(model, data)
        viewer.sync()
        log.append((data.time, q, dq, data.ctrl.copy()))

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print(
    f"final time={data.time:.3f}s, "
    f"q={data.qpos[0]:.4f}, "
    f"dq={data.qvel[0]:.4f}, "
    f"ctrl={np.array2string(data.ctrl, precision=3)}"
)
