"""阶段一验收 1：静止站立 30 秒。

流程：官方 rest 姿态 spawn → 3 秒沉降 → 30 秒舵机保持 q=0（平衡站姿）。
验收：水平漂移 < 20mm、高度波动 < 10mm、roll/pitch 峰值 < 5°。

用法：
    uv run python 7.quaddle-mjcf/run_stand.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "quaddle.xml"

SETTLE_S = 3.0
STAND_S = 30.0
THRESH_DRIFT_MM = 20.0
THRESH_DZ_MM = 10.0
THRESH_TILT_DEG = 5.0


def rpy_from_mat(mat: np.ndarray) -> tuple[float, float]:
    roll = np.degrees(np.arctan2(mat[2, 1], mat[2, 2]))
    pitch = np.degrees(np.arctan2(-mat[2, 0], np.hypot(mat[2, 1], mat[2, 2])))
    return float(roll), float(pitch)


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = 0.0

    for _ in range(int(SETTLE_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
    p0 = data.body("body").xpos.copy()

    worst = {"dz": 0.0, "roll": 0.0, "pitch": 0.0}
    for i in range(int(STAND_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if (i + 1) % 100 == 0:
            p = data.body("body").xpos
            mat = data.body("body").xmat.reshape(3, 3)
            roll, pitch = rpy_from_mat(mat)
            worst["dz"] = max(worst["dz"], abs(float(p[2] - p0[2])) * 1000)
            worst["roll"] = max(worst["roll"], abs(roll))
            worst["pitch"] = max(worst["pitch"], abs(pitch))

    p1 = data.body("body").xpos.copy()
    drift = float(np.hypot(*(p1[:2] - p0[:2])) * 1000)

    print(f"站立 {STAND_S:.0f}s：水平漂移 {drift:.1f}mm  高度波动 {worst['dz']:.1f}mm  "
          f"roll峰 {worst['roll']:.2f}°  pitch峰 {worst['pitch']:.2f}°  终高 {p1[2]*1000:.1f}mm")
    passed = (
        drift < THRESH_DRIFT_MM
        and worst["dz"] < THRESH_DZ_MM
        and worst["roll"] < THRESH_TILT_DEG
        and worst["pitch"] < THRESH_TILT_DEG
    )
    print("PASS ✅" if passed else "FAIL ❌")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
