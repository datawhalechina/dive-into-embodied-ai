"""诊断：单条 collect_v5 episode 的行为 trace（2s 间隔打印 bearing/dist/skill）。"""
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco
from collect_v5 import (DT_CTRL, SUB, build_scene, random_layout, wave_at,
                        SIGNS, BIAS, AMPS, PHASE0, FREQ, DUTY, STEER_LIMIT,
                        SIDE, CTRL_MAX, SETTLE_S, RAMP_S, COLORS, XML)

rng = np.random.default_rng(5000)
scene = HERE / "outputs" / "dataset" / "scene.xml"
build_scene(scene)
model = mujoco.MjModel.from_xml_path(str(scene))
model.opt.timestep = 0.001
data = mujoco.MjData(model)

# 复现 ep001 的随机流：跳过前 1 次布局/yaw 抽样（对齐 collect 顺序太繁琐，直接固定 rear 场景）
layout = [(0.29, 0.0), (0.5, 0.3), (-0.4, -0.2)]
color = "green"
tgt_xy = layout[1]
yaw0 = math.atan2(tgt_xy[1], tgt_xy[0]) - math.pi / 2  # 正后方

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in COLORS}

data.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
for i, c in enumerate(COLORS):
    model.geom_pos[geom_ids[c]][:2] = list(layout[i])
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
for _ in range(int(SETTLE_S / 0.001)):
    mujoco.mj_step(model, data)

amp_scale, phases_t = 0.0, 0.0
skill = skill0 = "forward"
last_switch_t, steer_now = 0.0, 0.0
t_sim = 0.0
DEADZONE = math.radians(25.0)

while t_sim < 45.0:
    r = data.body(body_id).xmat.reshape(3, 3)
    pos = data.body(body_id).xpos
    d_world = np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
    dist = float(np.linalg.norm(d_world[:2]))
    d_body = r.T @ d_world
    bearing = math.atan2(d_body[0], -d_body[1])

    if dist < 0.12:
        skill = "stop"
    elif abs(bearing) > DEADZONE:
        skill = "steer_L" if bearing > 0 else "steer_R"
    else:
        skill = "forward"
    if skill != skill0 and t_sim - last_switch_t < 0.4 and skill != "stop":
        skill = skill0
    steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
    if skill != skill0:
        last_switch_t = t_sim
        skill0 = skill

    amp_scale = min(1.0, amp_scale + DT_CTRL / RAMP_S)
    phases_t += 2 * math.pi * FREQ * DT_CTRL
    duty = np.full(4, DUTY)
    if steer_now > 0:
        duty[list(SIDE["L"])] *= (1 - STEER_LIMIT)
        duty[list(SIDE["R"])] *= (1 + STEER_LIMIT)
    elif steer_now < 0:
        duty[list(SIDE["L"])] *= (1 + STEER_LIMIT)
        duty[list(SIDE["R"])] *= (1 - STEER_LIMIT)
    w = wave_at(phases_t + PHASE0, duty)
    data.ctrl[:] = np.clip(SIGNS * (BIAS + AMPS * amp_scale * w), -CTRL_MAX, CTRL_MAX)

    for _ in range(5):
        for _ in range(SUB):
            mujoco.mj_step(model, data)
        t_sim += DT_CTRL

    if abs(t_sim / 2 - round(t_sim / 2)) < DT_CTRL / 2:
        r = data.body(body_id).xmat.reshape(3, 3)
        d_body2 = r.T @ d_world
        b2 = math.degrees(math.atan2(d_body2[0], -d_body2[1]))
        print(f"t={t_sim:5.1f} dist={dist*100:5.1f}cm bearing={b2:+6.1f}° skill={skill} "
              f"vx={-float((r.T @ data.qvel[:3])[1])*1000:5.1f}mm/s", flush=True)
    if dist < 0.12:
        print(f"到达 t={t_sim:.1f}s")
        break
os._exit(0)
