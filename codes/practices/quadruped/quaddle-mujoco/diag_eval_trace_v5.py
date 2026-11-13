"""诊断 3：ep0（seed 7000）评估轨迹逐 2s 打印 bearing/dist/skill/原始输出。"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "gait.json").read_text())
DT_PHYS, DT_CTRL, SUB = 0.001, 0.02, 20
SETTLE_S = 2.0
RAMP_S = float(CONFIG.get("ramp_seconds", 1.5))
CTRL_MAX = 1.2
SIGNS = np.array(CONFIG["actuator_signs"])
BIAS = np.array(CONFIG["bias_rad"])
AMPS = np.array(CONFIG["amplitude_rad"])
PHASE0 = np.array(CONFIG["phase_rad"])
FREQ = float(CONFIG["frequency_hz"])
DUTY = float(CONFIG["sweep_fraction"])
STEER_LIMIT = float(CONFIG["steering_limit"])
SIDE = {"L": (0, 3), "R": (1, 2)}

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
import mujoco


def wave_at(ph, duty):
    cycle = (ph / (2 * math.pi)) % 1.0
    phased = np.where(cycle < duty, math.pi * cycle / duty,
                      math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(phased)


scene_xml = HERE / "outputs" / "dataset_w0" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene_xml))
model.opt.timestep = DT_PHYS
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
            for c in ("red", "green", "blue")}

rng = np.random.default_rng(7000)
for ep in range(1):
    pts = []
    while len(pts) < 3:
        r = rng.uniform(0.25, 0.55)
        th = rng.uniform(-math.pi, math.pi)
        p = (r * math.cos(th), r * math.sin(th))
        if all(math.dist(p, q) >= 0.22 for q in pts):
            pts.append(p)
    yaw0 = float(rng.uniform(-math.pi, math.pi))
layout = pts
tgt_xy = np.array(layout[0])
color = "red"

mujoco.mj_resetData(model, data)
data.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
for i, c in enumerate(("red", "green", "blue")):
    model.geom_pos[geom_ids[c]][:2] = list(layout[i])
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
for _ in range(int(SETTLE_S / DT_PHYS)):
    mujoco.mj_step(model, data)

ckpt = str(HERE / "outputs" / "smolvla_quaddle_v5" / "checkpoints" / "last" / "pretrained_model")
cfg = PreTrainedConfig.from_pretrained(ckpt)
policy = SmolVLAPolicy.from_pretrained(ckpt)
policy.eval()
pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device="cuda")
renderer = mujoco.Renderer(model, height=160, width=160)

servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
amp_scale, phases_t = 0.0, 0.0
chunk, chunk_i = None, 0
skill0, last_switch_t = None, 0.0
steer_now = 0.0
t = 0.0
print(f"yaw0={math.degrees(yaw0):.0f}° tgt={tgt_xy}")
while t < 60.0:
    r = data.body(body_id).xmat.reshape(3, 3)
    pos = data.body(body_id).xpos
    dist = float(np.linalg.norm(tgt_xy - pos[:2]))

    if chunk is None or chunk_i >= 25:
        imgs = {}
        for cam in ("overhead", "front", "oblique"):
            renderer.update_scene(data, camera=cam)
            imgs[cam] = renderer.render().copy()
        state = np.zeros(8, dtype=np.float32)
        state[:4] = [data.qpos[servo_qposadr[i]] for i in range(4)]
        v_body = r.T @ data.qvel[:3]
        state[4] = float(-v_body[1])
        state[5] = float((r.T @ data.qvel[3:6])[2])
        d_body = r.T @ np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
        state[6] = math.atan2(d_body[0], -d_body[1])
        state[7] = dist
        batch = {
            "observation.images.camera1": torch.from_numpy(imgs["overhead"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
            "observation.images.camera2": torch.from_numpy(imgs["front"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
            "observation.images.camera3": torch.from_numpy(imgs["oblique"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [f"walk to the {color} block"],
        }
        batch = pre(batch)
        with torch.no_grad():
            chunk = policy.select_action(batch)
        chunk_i = 0

    a = chunk[0].float().cpu().numpy()
    chunk_i += 1
    vx, wz = float(a[0]), float(a[1])
    if vx < 0:
        vx = 0.0
    if abs(wz) > 0.5:
        skill = "steer_L" if wz > 0 else "steer_R"
    elif max(abs(vx), abs(wz)) < 0.12:
        skill = "stop"
    elif abs(wz) >= abs(vx) and abs(wz) > 0.15:
        skill = "steer_L" if wz > 0 else "steer_R"
    else:
        skill = "forward" if vx > 0 else "stop"
    if skill != skill0 and skill0 is not None and t - last_switch_t < 0.4 and skill != "stop":
        skill = skill0
    steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
    if skill != skill0:
        skill0, last_switch_t = skill, t

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
        t += DT_CTRL

    if abs(t / 2 - round(t / 2)) < DT_CTRL / 2:
        r = data.body(body_id).xmat.reshape(3, 3)
        d_body = r.T @ np.array([tgt_xy[0] - data.body(body_id).xpos[0], tgt_xy[1] - data.body(body_id).xpos[1], 0.0])
        b2 = math.degrees(math.atan2(d_body[0], -d_body[1]))
        wz_body = float((r.T @ data.qvel[3:6])[2])
        print(f"t={t:5.1f} dist={dist*100:5.1f}cm bearing={b2:+7.1f}° skill={skill:8s} "
              f"raw=({vx:+.2f},{wz:+.2f}) wz_body={wz_body:+.2f}rad/s ncon={data.ncon}", flush=True)
    if dist < 0.12:
        print(f"到达 t={t:.1f}s")
        break
os._exit(0)
