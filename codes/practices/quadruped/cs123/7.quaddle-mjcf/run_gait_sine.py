"""阶段一验收 2：正弦步态扫描。

从平衡站姿出发，对四舵机施加相位差正弦/偏斜正弦激励，测量净位移、
偏航与最低离地高度。球面足端 + 各向同性摩擦下对称激励只会产生漂移爬行——
本脚本量化这个基线，为阶段二（官方步态回放 / RL）提供对照。

用法：
    uv run python 7.quaddle-mjcf/run_gait_sine.py            # 全部 6 组
    uv run python 7.quaddle-mjcf/run_gait_sine.py --quick    # 前 3 组
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "quaddle.xml"
SETTLE_S = 2.0
GAIT_S = 8.0
TILT_LIMIT_DEG = 30.0  # roll/pitch 超过视为翻倒
SERVO_TAU = 0.03  # 舵机一阶响应时间常数（s），约 5.3 Hz 截止

WAVES = {
    "sine": lambda s: np.sin(s),
    "saw+": lambda s: np.sin(s) + 0.45 * np.sin(2 * s),   # 快摆慢回
    "saw-": lambda s: np.sin(s) - 0.45 * np.sin(2 * s),   # 慢摆快回
}


def rpy(mat: np.ndarray) -> tuple[float, float]:
    roll = np.degrees(np.arctan2(mat[2, 1], mat[2, 2]))
    pitch = np.degrees(np.arctan2(-mat[2, 0], np.hypot(mat[2, 1], mat[2, 2])))
    return float(roll), float(pitch)


def run_gait(model, data, wave_name: str, amp_deg: float, freq_hz: float,
             phases, gait_s: float = GAIT_S) -> dict:
    wave = WAVES[wave_name]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    dt = model.opt.timestep
    x0 = None
    z_min = np.inf
    servo_pos = np.zeros(4)
    for i in range(int((SETTLE_S + gait_s) / dt)):
        t = i * dt
        if t < SETTLE_S:
            data.ctrl[:] = 0.0
        else:
            if x0 is None:
                x0 = data.body("body").xpos.copy()
            ts = t - SETTLE_S
            alpha = 1.0 - np.exp(-dt / SERVO_TAU)
            for k in range(4):
                target = np.radians(amp_deg) * wave(2 * np.pi * freq_hz * ts + phases[k])
                # 舵机一阶响应：限制角速度冲击，防 QACC 爆炸（真实舵机带宽有限）
                servo_pos[k] += alpha * (np.clip(target, -2.44, 2.44) - servo_pos[k])
            data.ctrl[:] = servo_pos
        mujoco.mj_step(model, data)
        z_min = min(z_min, float(data.body("body").xpos[2]))
    p1 = data.body("body").xpos.copy()
    mat = data.body("body").xmat.reshape(3, 3).copy()
    roll, pitch = rpy(mat)
    dxy = (p1[:2] - x0[:2]) * 1000.0
    speed = float(np.hypot(dxy[0], dxy[1])) / gait_s
    return {
        "dx_mm": float(dxy[0]),
        "dy_mm": float(dxy[1]),
        "speed_mms": speed,
        "roll_deg": roll,
        "pitch_deg": pitch,
        "z_min_mm": float(z_min) * 1000.0,
        "fell": abs(roll) > TILT_LIMIT_DEG or abs(pitch) > TILT_LIMIT_DEG,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="只跑前 3 组")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    trot = [0.0, np.pi, 0.0, np.pi]                       # 对角步态
    wave = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]         # 波次步态
    cases = [
        ("sine", 8.0, 1.5, trot),
        ("sine", 10.0, 1.5, wave),
        ("saw+", 8.0, 1.5, trot),
        ("saw+", 10.0, 1.5, wave),
        ("saw-", 8.0, 1.5, trot),
        ("saw-", 10.0, 1.5, wave),
    ]
    if args.quick:
        cases = cases[:3]

    print(f"{'波形':>5s} {'幅':>5s} {'频':>6s} {'相位':>4s} | "
          f"{'dx(mm)':>8s} {'dy(mm)':>8s} {'速度':>9s} {'roll':>6s} {'pitch':>6s} {'最低高':>6s} 状态")
    for wname, amp, freq, ph in cases:
        r = run_gait(model, data, wname, amp, freq, ph)
        ph_name = "trot" if ph is trot else "wave"
        status = "翻倒" if r["fell"] else "ok"
        print(f"{wname:>5s} {amp:4.0f}° {freq:5.1f}Hz {ph_name:>4s} | "
              f"{r['dx_mm']:7.1f} {r['dy_mm']:7.1f} {r['speed_mms']:7.1f}mm/s "
              f"{r['roll_deg']:5.1f}° {r['pitch_deg']:5.1f}° {r['z_min_mm']:5.0f}mm {status}")


if __name__ == "__main__":
    main()
