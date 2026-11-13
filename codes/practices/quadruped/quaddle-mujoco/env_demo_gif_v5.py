"""v5 环境演示 GIF：《Quaddle 入门》环境 + VLA 任务场景（3 积木）。

序列：站立 → 前进 → 原地转向（duty 不对称，v5 转向原语）→ 前进 → 停止。
场景复用 worker 数据集的 scene.xml（含积木与相机），相机用 MjvCamera 跟拍。
"""
import json
import math
import os
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
SCENE = HERE / "outputs" / "dataset_w0" / "scene.xml"
CONFIG = json.loads((HERE / "gait.json").read_text())

DT_PHYS, DT_CTRL, SUB = 0.001, 0.02, 20
RAMP_S = float(CONFIG.get("ramp_seconds", 1.5))
CTRL_MAX = 1.2
SIGNS = __import__("numpy").array(CONFIG["actuator_signs"])
BIAS = __import__("numpy").array(CONFIG["bias_rad"])
AMPS = __import__("numpy").array(CONFIG["amplitude_rad"])
PHASE0 = __import__("numpy").array(CONFIG["phase_rad"])
FREQ = float(CONFIG["frequency_hz"])
DUTY = float(CONFIG["sweep_fraction"])
STEER_LIMIT = float(CONFIG["steering_limit"])
SIDE = {"L": (0, 3), "R": (1, 2)}
PHASES = [("stand", 1.5), ("forward", 3.0), ("steer_L", 2.5), ("forward", 2.0), ("stop", 1.0)]

import xml.etree.ElementTree as ET
import numpy as np


def wave_at(ph, duty):
    cycle = (ph / (2 * math.pi)) % 1.0
    phased = np.where(cycle < duty, math.pi * cycle / duty,
                      math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(phased)


model = mujoco.MjModel.from_xml_path(str(SCENE))
model.opt.timestep = DT_PHYS
data = mujoco.MjData(model)
body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
for _ in range(int(2.0 / DT_PHYS)):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=240, width=320)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.25, 0.0, 0.05]
cam.distance = 0.85
cam.azimuth = 140
cam.elevation = -32

amp_scale, phases_t = 0.0, 0.0
frames = []

for skill, dur in PHASES:
    steps = int(dur / DT_CTRL)
    steer = 1.0 if skill == "steer_L" else 0.0
    for _ in range(steps):
        amp_scale = min(1.0, amp_scale + DT_CTRL / RAMP_S)
        phases_t += 2 * math.pi * FREQ * DT_CTRL
        duty = np.full(4, DUTY)
        if steer > 0:
            duty[list(SIDE["L"])] *= (1 - steer * STEER_LIMIT)
            duty[list(SIDE["R"])] *= (1 + steer * STEER_LIMIT)
        w = wave_at(phases_t + PHASE0, duty)
        amps = AMPS * (0.0 if skill in ("stand", "stop") else amp_scale)
        data.ctrl[:] = np.clip(SIGNS * (BIAS + amps * w), -CTRL_MAX, CTRL_MAX)
        for _ in range(SUB):
            mujoco.mj_step(model, data)
        # 10Hz 采帧
        if int(phases_t / (2 * math.pi * FREQ) / 0.1) != int((phases_t - 2 * math.pi * FREQ * DT_CTRL) / (2 * math.pi * FREQ) / 0.1):
            p = data.body(body).xpos
            cam.lookat[:] = [p[0] * 0.5 + 0.15, p[1] * 0.5, 0.05]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

out = str(HERE / "outputs" / "env_demo_v5.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)
import imageio.v2 as imageio
imageio.mimsave(out, frames, fps=10, loop=0)
print("GIF_OK", out, len(frames))
os._exit(0)
