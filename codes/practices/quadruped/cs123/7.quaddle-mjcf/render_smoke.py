"""阶段一验收 3：EGL 无头渲染烟测。

离屏渲染一帧站立姿态，验证 MUJOCO_GL=egl 的渲染链路（VLA 数据采集的前置条件）。

用法：
    MUJOCO_GL=egl uv run python 7.quaddle-mjcf/render_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "quaddle.xml"
OUT_PNG = LAB_DIR / "render_smoke.png"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = 0.0
    for _ in range(int(2.0 / model.opt.timestep)):  # 沉降到站姿
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=360, width=480)
    renderer.update_scene(data, camera="tracking")
    frame = renderer.render()
    out = np.asarray(frame)
    assert out.shape[:2] == (360, 480), f"意外尺寸 {out.shape}"
    assert out.std() > 5.0, "渲染全黑？"
    try:
        import imageio.v2 as imageio
        imageio.imwrite(OUT_PNG, out)
        print(f"渲染 OK：{OUT_PNG}（{out.shape[1]}x{out.shape[0]}，std={out.std():.1f}）")
    except ImportError:
        print(f"渲染 OK（未安装 imageio，跳过保存）：std={out.std():.1f}")


if __name__ == "__main__":
    main()
