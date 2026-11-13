"""v5 环境迁移验证：在《Quaddle 入门》环境（quaddle-mujoco）扫描转向原语。

背景：VLA 管线从 cs123/7.quaddle-mjcf 迁移到本环境（用户 2026-11-11 决策，
环境统一）。cs123 版的结论（滚轮单向棘轮 → 支点转是唯一可靠转向原语、
差频弧线档不可靠）需在本环境重新验证——地面摩擦（0.8 vs 旧 1.0+）、
滚轮阻尼、cosine_duty 波形均不同。

候选（沿用本环境 gait.json 的驱动格式：signs/bias/amplitude/cosine_duty）：
  base_walk       原生 coordinated_crawl（前进基线）
  steer_L         原生转向机制：左右幅值不对称（heading_kp 拉满 steer_limit）
  pivot_L/R       一侧幅值=0（保留 bias 静息角）+ 另侧行驶 —— cs123 版的 spin 机制
  steer_pivot_L   转向机制拉满 + 行驶侧再不对称（混合）

判定同 v4：net_yaw 显著且 |vx| 小 → 可用 spin；yaw 变化但 vx 大 → 紧弧线。
"""

from __future__ import annotations

import json
import math
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
L_DUTY, R_DUTY = 0, 1  # left_front=0, right_front=1（rb/lb 与同侧同参）
SIDE = {"L": (0, 3), "R": (1, 2)}

# 幅值缩放：1=原值；0=停摆（保留 bias 静息角）
CANDIDATES = {
    "steer_L": {"scale_L": 1.0, "scale_R": 1.0, "steer": float(CONFIG["steering_limit"])},
    "steer_R": {"scale_L": 1.0, "scale_R": 1.0, "steer": -float(CONFIG["steering_limit"])},
    "steer_R_a33": {"scale_L": 1.0, "scale_R": 1.0, "steer": -0.33},
    "steer_R_a45": {"scale_L": 1.0, "scale_R": 1.0, "steer": -0.45},
    "steer_L_a33": {"scale_L": 1.0, "scale_R": 1.0, "steer": 0.33},
    "pivot_L": {"scale_L": 0.0, "scale_R": 1.0, "steer": 0.0},
    "pivot_R": {"scale_L": 1.0, "scale_R": 0.0, "steer": 0.0},
}


def wave_at(t: float, duty: float):
    """cosine_duty 波形：占空比不对称的 cos（推进之源）。"""
    cycle = (t / (2 * math.pi)) % 1.0
    ph = np.where(cycle < duty, math.pi * cycle / duty,
                  math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(ph)


def run(model, data, name: str, p: dict, seconds=10.0):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(SETTLE_S / DT_PHYS)):
        mujoco.mj_step(model, data)

    r0 = data.body(body).xmat.reshape(3, 3).copy()
    p0 = data.body(body).xpos.copy()
    yaw0 = math.atan2(r0[1, 0], r0[0, 0])

    steer = p["steer"]
    scales = np.ones(4)
    scales[list(SIDE["L"])] = p["scale_L"]
    scales[list(SIDE["R"])] = p["scale_R"]
    duty = np.array([DUTY, DUTY, DUTY, DUTY])
    if steer > 0:      # 原生机制：左转 = 左侧行程缩、右侧行程放
        duty[list(SIDE["L"])] *= (1 - steer)
        duty[list(SIDE["R"])] *= (1 + steer)
    else:
        duty[list(SIDE["L"])] *= (1 + abs(steer))
        duty[list(SIDE["R"])] *= (1 - abs(steer))

    amp_scale = 0.0
    fallen = False
    wz_sum = vx_sum = n = 0
    max_roll = 0.0
    for step in range(int(seconds / DT_CTRL)):
        amp_scale = min(1.0, amp_scale + DT_CTRL / RAMP_S)
        t = step * DT_CTRL
        ph = 2 * math.pi * FREQ * t + PHASE0
        w = wave_at(ph, duty)
        amps = AMPS * scales * amp_scale * (1 + (steer if steer > 0 else 0) * SIGNS * 0)  # 转向已入 duty
        target = np.clip(SIGNS * (BIAS + amps * w), -CTRL_MAX, CTRL_MAX)
        data.ctrl[:] = target
        for _ in range(SUB):
            mujoco.mj_step(model, data)
        r = data.body(body).xmat.reshape(3, 3)
        v_body = r.T @ data.qvel[:3]
        vx_sum += float(-v_body[1])
        wz_sum += float((r.T @ data.qvel[3:6])[2])
        n += 1
        roll = abs(math.atan2(r[2, 1], r[2, 2]))
        max_roll = max(max_roll, roll)
        if data.body(body).xpos[2] < 0.030 or roll > 0.52:
            fallen = True
            break

    r1 = data.body(body).xmat.reshape(3, 3).copy()
    p1 = data.body(body).xpos.copy()
    yaw1 = math.atan2(r1[1, 0], r1[0, 0])
    dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    fwd = float(np.dot((p1[:2] - p0[:2]), r0[:2, 0]))
    lat = float(np.dot((p1[:2] - p0[:2]), r0[:2, 1]))
    return dict(name=name, fwd_mm=fwd * 1e3, lat_mm=lat * 1e3, yaw_deg=math.degrees(dyaw),
                vx=vx_sum / n * 1e3, wz=wz_sum / n, roll=math.degrees(max_roll), fallen=fallen)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = DT_PHYS
    data = mujoco.MjData(model)
    hdr = (f"{'candidate':<15} {'fwd(mm)':>8} {'lat(mm)':>8} {'yaw(°)':>7} "
           f"{'vx(mm/s)':>9} {'wz(rad/s)':>9} {'roll°':>6} fallen")
    print(hdr)
    print("-" * len(hdr))
    for name, p in CANDIDATES.items():
        m = run(model, data, name, p)
        print(f"{m['name']:<15} {m['fwd_mm']:>8.1f} {m['lat_mm']:>8.1f} {m['yaw_deg']:>7.1f} "
              f"{m['vx']:>9.1f} {m['wz']:>9.3f} {m['roll']:>6.1f} {'YES' if m['fallen'] else 'no'}",
              flush=True)
    print("\n判定：yaw 显著且 |vx| 小 → spin 档可用；yaw 变化但 vx 大 → 紧弧线。")
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
