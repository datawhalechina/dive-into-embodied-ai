"""v5 专家闭环 smoke：steer 转向原语（duty 不对称）在 rear 场景的裁决。

继承 v4 方法论（expert_spin_smoke.py）：目标固定正后方的场景里验证专家能否
到达。v5 专家设计（基于 spin_feasibility_v5.py 实测：steer_L wz+1.2 rad/s、
转弯半径 2.6cm、10s 不翻）：

    dist < 12cm                 → stop
    |bearing| > deadzone        → steer ±1（全幅 duty 不对称，方向锁定）
    否则                        → forward（steer=0）

动作空间回退 2 维 [vx_dir, wz_dir]：wz ∈ {-1,0,1} 映射执行器 steer 值
（flow matching 二值档位，v4 已验证可学）。切档驻留 0.4s 保留。

场景：2/3 rear（目标正后方 ±30°）+ 1/3 前侧方（±60°，验证不回退）。
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
XML = HERE / "models" / "scene.xml"
CONFIG = json.loads((HERE / "gait.json").read_text())

DT_PHYS, DT_CTRL = 0.001, 0.02
SUB = 20
SETTLE_S = float(CONFIG.get("settle_seconds", 2.0))
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

COLORS = ("red", "green", "blue")
COLOR_RGBA = {"red": (0.9, 0.2, 0.2, 1), "green": (0.2, 0.8, 0.3, 1), "blue": (0.25, 0.4, 0.9, 1)}
OBJ_RNG = (0.28, 0.70)
MIN_GAP = 0.22
SUCCESS_DIST = 0.12
TIMEOUT_S = 45.0
DWELL_S = 0.4


def wave_at(ph, duty):
    cycle = (ph / (2 * math.pi)) % 1.0
    phased = np.where(cycle < duty, math.pi * cycle / duty,
                      math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(phased)


def build_scene(out_xml: Path) -> None:
    """scene.xml 副本 + 注入 3 积木 + overhead 相机（mesh 相对路径不变，同目录写出）。"""
    tree = ET.parse(XML)
    world = tree.getroot().find("worldbody")
    for c in COLORS:
        ET.SubElement(world, "geom", name=f"block_{c}", type="box",
                      size="0.025 0.025 0.025",
                      rgba=" ".join(f"{v:g}" for v in COLOR_RGBA[c]),
                      pos="0 0 0.025", contype="1", conaffinity="1")
    ET.SubElement(world, "camera", name="overhead", pos="0 0 1.4", xyaxes="1 0 0 0 1 0")
    tree.write(out_xml, encoding="unicode")


def random_layout(rng):
    for _ in range(200):
        pts = []
        while len(pts) < 3:
            r = rng.uniform(*OBJ_RNG)
            th = rng.uniform(-math.pi, math.pi)
            p = (r * math.cos(th), r * math.sin(th))
            if all(math.dist(p, q) >= MIN_GAP for q in pts):
                pts.append(p)
        return pts
    raise RuntimeError


def run_episode(model, data, body_id, geom_ids, layout, target_xy, yaw0, deadzone, expert="v5"):
    steer_active_dir = 0.0     # 0=无转向；±1=方向锁定
    last_switch_t = 0.0
    skill0 = None
    amp_scale = 0.0
    phases_t = 0.0

    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
    for i, c in enumerate(COLORS):
        model.geom_pos[geom_ids[c]][:2] = list(layout[i])
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(SETTLE_S / DT_PHYS)):
        mujoco.mj_step(model, data)

    tgt = np.array([target_xy[0], target_xy[1], 0.0])
    t, skill = 0.0, "forward"
    success = fallen = False
    trace = []
    steer_now = 0.0
    while t < TIMEOUT_S:
        r = data.body(body_id).xmat.reshape(3, 3)
        pos = data.body(body_id).xpos
        d_world = tgt - pos
        dist = float(np.linalg.norm(d_world[:2]))
        d_body = r.T @ d_world
        bearing = math.atan2(d_body[0], -d_body[1])  # 左正

        if dist < SUCCESS_DIST:
            skill = "stop"
            success = True
        elif abs(bearing) > deadzone:
            skill = "steer_L" if bearing > 0 else "steer_R"
        else:
            skill = "forward"

        if skill != skill0 and skill0 is not None and t - last_switch_t < DWELL_S and skill != "stop":
            skill = skill0
        if skill == "steer_L":
            steer_now = 1.0
        elif skill == "steer_R":
            steer_now = -1.0
        else:
            steer_now = 0.0

        amp_scale = min(1.0, amp_scale + DT_CTRL / RAMP_S)
        phases_t += 2 * math.pi * FREQ * DT_CTRL
        duty = np.full(4, DUTY)
        if steer_now > 0:
            duty[list(SIDE["L"])] *= (1 - steer_now * STEER_LIMIT)
            duty[list(SIDE["R"])] *= (1 + steer_now * STEER_LIMIT)
        elif steer_now < 0:
            duty[list(SIDE["L"])] *= (1 + abs(steer_now) * STEER_LIMIT)
            duty[list(SIDE["R"])] *= (1 - abs(steer_now) * STEER_LIMIT)
        w = wave_at(2 * math.pi * FREQ * 0 + phases_t + PHASE0, duty)
        amps = AMPS * amp_scale
        data.ctrl[:] = np.clip(SIGNS * (BIAS + amps * w), -CTRL_MAX, CTRL_MAX)
        for _ in range(SUB):
            mujoco.mj_step(model, data)
        t += DT_CTRL

        if skill != skill0:
            r_now = data.body(body_id).xmat.reshape(3, 3)
            d_body2 = r_now.T @ np.array([tgt[0] - data.body(body_id).xpos[0],
                                          tgt[1] - data.body(body_id).xpos[1], 0.0])
            b_now = math.atan2(d_body2[0], -d_body2[1])
            trace.append((t, math.degrees(b_now), float(np.linalg.norm((tgt - data.body(body_id).xpos)[:2])), skill))
            skill0 = skill
            last_switch_t = t

        if data.body(body_id).xpos[2] < 0.030:
            fallen = True
            break
        if success:
            break
    return dict(success=success, fallen=fallen, t=t, dist=dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=4100)
    args = ap.parse_args()

    scene = XML.parent / "_scene_vla_tmp.xml"  # 与 models/ 同目录，mesh 相对路径才可解析
    build_scene(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = DT_PHYS
    data = mujoco.MjData(model)
    scene.unlink()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in COLORS}

    rng = np.random.default_rng(args.seed)
    tally = {"v5": []}
    print(f"{'scene':<6} {'结果':<6} {'用时':>6} {'末距(cm)':>8}  切档序列")
    print("-" * 60)
    for si in range(args.scenes):
        layout = random_layout(rng)
        color = COLORS[si % 3]
        tgt_xy = layout[COLORS.index(color)]
        th_t = math.atan2(tgt_xy[1], tgt_xy[0])
        if si < args.scenes * 2 // 3:
            yaw0 = th_t - math.pi / 2 + rng.uniform(-math.radians(30), math.radians(30))
        else:
            yaw0 = th_t + rng.uniform(-math.radians(60), math.radians(60))
        dz = rng.uniform(math.radians(12), math.radians(40))
        m = run_episode(model, data, body_id, geom_ids, layout, tgt_xy, yaw0, dz)
        tag = "✓到达" if m["success"] else ("✗翻倒" if m["fallen"] else "✗超时")
        tally["v5"].append(tag)
        print(f"{si:<6} {tag:<6} {m['t']:>6.1f} {m['dist']*100:>8.1f}", flush=True)
    ok = sum(1 for x in tally["v5"] if "✓" in x)
    fall = sum(1 for x in tally["v5"] if "翻" in x)
    print(f"\nv5_steer: 到达 {ok}/{len(tally['v5'])}  翻倒 {fall}")
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
