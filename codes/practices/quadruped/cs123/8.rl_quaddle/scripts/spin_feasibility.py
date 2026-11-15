"""原地转（spin/pivot）原语可行性扫描——W5 改进路线 1 的前置验证。

背景（quaddle-data/README W5 根因）：滚轮单向棘轮 + 差频转向必带直行分量，
弧线稳态中 bearing 恒定 → 目标近正后方（bearing>120°）无限绕弧。动作空间缺
"纯原地转"原语。本脚本在滚轮足端模型上扫描候选步态表，回答：

    Q1 支点转（一侧腿 amp=0 停摆，另一侧 trot）能否产生 |vx|≈0 的净旋转？
    Q2 左右反相（同频同幅、相位差 π）是否产生偏航（还是仍直线前进）？
    Q3 各候选的稳定性（roll/pitch、翻倒）与转向速率量级？

判定标准：net_yaw > 60°/10s（≥0.105 rad/s）且 |vx_mean| < 20 mm/s → 可用 spin 档；
净旋转存在但带直行 → 退化为紧弧线档（也能破死锁，但转向效率低）。

运行（工作台/本地均可，仅需 mujoco+numpy，无需渲染）：
    python spin_feasibility.py [--seconds 10] [--xml PATH]

腿序 lf, rf, rb, lb；左腿 = {0: lf, 3: lb}，右腿 = {1: rf, 2: rb}
（home keyframe 前进=世界 +x，CPG 惯例 turn_left → wz>0，右侧行驶更快）。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

LAB_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = LAB_DIR / ".." / "7.quaddle-mjcf" / "quaddle.xml"

DT_PHYSICS = 0.001
DT_CONTROL = 0.02
N_SUBSTEPS = 20
SETTLE_S = 1.2
AMP_RAMP_S = 0.3
CTRL_LIMIT = 2.44
FALLEN_Z = 0.030
FALLEN_ANGLE = 0.52  # ≈30°

TROT = (0.0, math.pi / 2, 0.0, math.pi / 2)  # lf, rf, rb, lb


def leg(amp_deg: float, freq: float, ph0: float, skew: float = -0.45):
    return (amp_deg, freq, ph0, skew)


# 候选步态表：(名字, [(amp_deg, freq, phase0, skew)] × 4)
CANDIDATES = {
    # -- 基线（标定过的行为，做对照）--
    "base_forward": [leg(10.5, 2.5, p) for p in TROT],
    "base_turn_L_strong": [
        leg(10.5, 1.0, 0.0), leg(10.5, 3.0, TROT[1]),
        leg(10.5, 1.0, 0.0), leg(10.5, 3.0, TROT[3]),
    ],
    # -- Q1 支点转：一侧停摆 --
    "pivot_L_a0": [  # 左腿停摆，右腿 trot → 期望 wz>0
        leg(0, 2.5, TROT[0]), leg(10.5, 2.5, TROT[1]),
        leg(10.5, 2.5, TROT[2]), leg(0, 2.5, TROT[3]),
    ],
    "pivot_R_a0": [  # 右腿停摆，左腿 trot → 期望 wz<0
        leg(10.5, 2.5, TROT[0]), leg(0, 2.5, TROT[1]),
        leg(0, 2.5, TROT[2]), leg(10.5, 2.5, TROT[3]),
    ],
    "pivot_L_a3": [  # 左腿 3° 轻摆（保留接触，减小拖曳冲击）
        leg(3, 2.5, TROT[0]), leg(10.5, 2.5, TROT[1]),
        leg(10.5, 2.5, TROT[2]), leg(3, 2.5, TROT[3]),
    ],
    "pivot_L_a0_f3": [  # 行驶侧 3.0Hz 提速
        leg(0, 2.5, TROT[0]), leg(10.5, 3.0, TROT[1]),
        leg(10.5, 3.0, TROT[2]), leg(0, 2.5, TROT[3]),
    ],
    # -- Q2 左右反相 --
    "antiphase_LR": [  # 左右相位差 π，同频同幅
        leg(10.5, 2.5, 0.0), leg(10.5, 2.5, math.pi),
        leg(10.5, 2.5, math.pi), leg(10.5, 2.5, 0.0),
    ],
    # -- 对角支点 --
    "diag_pivot_LF": [  # lf+rb 停摆，rf+lb trot（对角支点）
        leg(0, 2.5, TROT[0]), leg(10.5, 2.5, TROT[1]),
        leg(0, 2.5, TROT[2]), leg(10.5, 2.5, TROT[3]),
    ],
}


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def run_candidate(model, data, name: str, table, seconds: float) -> dict:
    """reset → 沉降 → 跑 skill → 返回净位移/净偏航/稳定性指标。"""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(SETTLE_S / DT_PHYSICS)):
        mujoco.mj_step(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

    def pose():
        r = data.body(body_id).xmat.reshape(3, 3)
        p = data.body(body_id).xpos
        fwd = r @ np.array([0.0, -1.0, 0.0])          # body 前向（世界系）
        yaw = math.atan2(fwd[1], fwd[0])
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
        return p.copy(), yaw, roll, pitch, r.copy()

    p0, yaw0, _, _, R0 = pose()
    phases = np.zeros(4)                               # 每腿独立相位钟
    amp_scale = 0.0
    fallen = False
    wz_sum, vx_sum, n = 0.0, 0.0, 0
    max_roll = max_pitch = 0.0

    n_steps = int(seconds / DT_CONTROL)
    for _ in range(n_steps):
        amp_scale = min(1.0, amp_scale + DT_CONTROL / AMP_RAMP_S)
        for k, (amp_deg, freq, ph0, skew) in enumerate(table):
            phases[k] += 2.0 * math.pi * freq * DT_CONTROL
            s = phases[k] + ph0
            target = math.radians(amp_deg * amp_scale) * (
                math.sin(s) + skew * math.sin(2.0 * s)
            )
            data.ctrl[k] = float(np.clip(target, -CTRL_LIMIT, CTRL_LIMIT))
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)

        r = data.body(body_id).xmat.reshape(3, 3)
        v_body = r.T @ data.qvel[:3]
        vx_sum += float(-v_body[1])
        wz_sum += float((r.T @ data.qvel[3:6])[2])
        n += 1
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
        max_roll, max_pitch = max(max_roll, abs(roll)), max(max_pitch, abs(pitch))
        if data.body(body_id).xpos[2] < FALLEN_Z or max_roll > FALLEN_ANGLE or max_pitch > FALLEN_ANGLE:
            fallen = True
            break

    p1, yaw1, _, _, _ = pose()
    d_world = p1[:2] - p0[:2]
    fwd0 = R0 @ np.array([0.0, -1.0, 0.0])             # 初期前向（世界系）
    right0 = np.array([-fwd0[1], fwd0[0]])             # 初期右侧（世界系）
    fwd_d = float(np.dot(d_world, fwd0[:2]))
    lat_d = float(np.dot(d_world, right0))
    return {
        "name": name,
        "fwd_mm": fwd_d * 1000.0,
        "lat_mm": lat_d * 1000.0,
        "yaw_deg": math.degrees(wrap_angle(yaw1 - yaw0)),
        "vx_mean_mm_s": vx_sum / n * 1000.0,
        "wz_mean": wz_sum / n,
        "max_roll_deg": math.degrees(max_roll),
        "max_pitch_deg": math.degrees(max_pitch),
        "fallen": fallen,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--xml", default=str(MODEL_PATH))
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = DT_PHYSICS
    data = mujoco.MjData(model)

    print(f"模型: {args.xml}  每候选 {args.seconds:.0f}s\n")
    hdr = (f"{'candidate':<20} {'fwd(mm)':>8} {'lat(mm)':>8} {'yaw(°)':>7} "
           f"{'vx(mm/s)':>9} {'wz(rad/s)':>9} {'roll°':>6} {'pitch°':>7} fallen")
    print(hdr)
    print("-" * len(hdr))
    for name, table in CANDIDATES.items():
        m = run_candidate(model, data, name, table, args.seconds)
        print(f"{m['name']:<20} {m['fwd_mm']:>8.1f} {m['lat_mm']:>8.1f} "
              f"{m['yaw_deg']:>7.1f} {m['vx_mean_mm_s']:>9.1f} {m['wz_mean']:>9.3f} "
              f"{m['max_roll_deg']:>6.1f} {m['max_pitch_deg']:>7.1f} "
              f"{'YES' if m['fallen'] else 'no'}", flush=True)

    print("\n判定：net_yaw>60° 且 |vx_mean|<20mm/s → 可用 spin 档；"
          "yaw 有净变化但 vx 大 → 紧弧线（次选）。")


if __name__ == "__main__":
    main()
