"""Quaddle 环境演示 GIF：站立 → 前进 → 原地转（v4 spin 原语）→ 停止。"""
import math
import os

import mujoco

QD = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(QD, "..", "..", "quadruped", "cs123", "7.quaddle-mjcf", "quaddle.xml")

DT_PHYS, DT_CTRL, SUB = 0.001, 0.02, 20
RAMP = 0.3
LIMIT = 2.44
TROT = (0.0, math.pi / 2, 0.0, math.pi / 2)

SKILLS = {
    "stand": [(0.0, 2.5, p, 0.0) for p in TROT],
    "forward": [(10.5, 2.5, p, -0.45) for p in TROT],
    "spin_left": [
        (0.0, 2.5, TROT[0], -0.45), (10.5, 2.5, TROT[1], -0.45),
        (10.5, 2.5, TROT[2], -0.45), (0.0, 2.5, TROT[3], -0.45),
    ],
    "stop": [(0.0, 2.5, 0.0, 0.0)] * 4,
}
PHASES = [("stand", 1.5), ("forward", 3.0), ("spin_left", 2.5), ("stop", 1.0)]

model = mujoco.MjModel.from_xml_path(XML)
model.opt.timestep = DT_PHYS
data = mujoco.MjData(model)
body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
for _ in range(int(1.2 / DT_PHYS)):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=240, width=320)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.05, 0.0, 0.05]
cam.distance = 0.55
cam.azimuth = 140
cam.elevation = -28
p = [0.0, 0.0, 0.0]  # lookat（世界系，home 朝向 +x）
phases = [0.0] * 4
amp = 0.0
frames = []

for skill, dur in PHASES:
    steps = int(dur / DT_CTRL)
    for _ in range(steps):
        amp = min(1.0, amp + DT_CTRL / RAMP) if SKILLS[skill][0][0] > 0 else min(1.0, amp + DT_CTRL / RAMP)
        for k, (a_deg, freq, ph0, skew) in enumerate(SKILLS[skill]):
            phases[k] += 2 * math.pi * freq * DT_CTRL
            s = phases[k] + ph0
            data.ctrl[k] = float(
                min(LIMIT, max(-LIMIT, math.radians(a_deg * amp) * (math.sin(s) + skew * math.sin(2 * s))))
            )
        for _ in range(SUB):
            mujoco.mj_step(model, data)
        p = data.body(body).xpos
        cam.lookat[:] = [p[0] + 0.05, p[1], 0.05]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

out = os.path.join(QD, "outputs", "env_demo.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)
import imageio.v2 as imageio
imageio.mimsave(out, frames, fps=12, loop=0)
print("GIF_OK", out, len(frames))
os._exit(0)
