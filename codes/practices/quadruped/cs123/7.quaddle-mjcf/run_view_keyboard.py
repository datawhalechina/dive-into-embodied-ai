"""交互查看器：WASD 控制四舵机步态。

在 MuJoCo 原生 viewer 中查看 Quaddle，键盘开环步态控制：
    W / S   加大 / 反向 摆动幅度（前进 / 后退倾向）
    A / D   加大 / 减小 左右侧幅度差（转向倾向）
    空格    站立（幅度清零）
    关闭窗口退出

需要图形环境（本机显示器或 X11 转发）；无头环境请用 render_smoke.py。
球面足端下方向控制是开环粗糙的——精确速度控制是阶段二（RL）的目标。

用法：
    uv run python 7.quaddle-mjcf/run_view_keyboard.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "quaddle.xml"

SETTLE_S = 2.0
BASE_AMP = math.radians(15.0)
BASE_FREQ = 2.0
SERVO_TAU = 0.03

# GLFW keycode
KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE = 87, 65, 83, 68, 32


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    state = {"amp": 0.0, "turn": 0.0}  # amp: -1..1 摆幅档位；turn: -1..1 转向档位

    def on_key(keycode: int) -> None:
        step = 0.1
        if keycode == KEY_W:
            state["amp"] = min(1.0, state["amp"] + step)
        elif keycode == KEY_S:
            state["amp"] = max(-1.0, state["amp"] - step)
        elif keycode == KEY_A:
            state["turn"] = min(1.0, state["turn"] + step)
        elif keycode == KEY_D:
            state["turn"] = max(-1.0, state["turn"] - step)
        elif keycode == KEY_SPACE:
            state["amp"] = 0.0
            state["turn"] = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        for _ in range(int(SETTLE_S / model.opt.timestep)):  # 沉降
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep
        servo_pos = np.zeros(4)
        phases = np.array([0.0, math.pi, 0.0, math.pi])  # trot
        alpha = 1.0 - math.exp(-dt / SERVO_TAU)
        while viewer.is_running():
            t = data.time - SETTLE_S
            amp = BASE_AMP * state["amp"]
            turn = BASE_AMP * state["turn"]
            s = 2.0 * math.pi * BASE_FREQ * t
            targets = np.array([
                amp * math.sin(s + phases[0]) + turn * math.sin(s),
                amp * math.sin(s + phases[1]) - turn * math.sin(s),
                amp * math.sin(s + phases[2]) + turn * math.sin(s),
                amp * math.sin(s + phases[3]) - turn * math.sin(s),
            ])
            servo_pos += alpha * (targets - servo_pos)
            data.ctrl[:] = servo_pos
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
