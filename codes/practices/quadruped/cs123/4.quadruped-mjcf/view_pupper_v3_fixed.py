"""Minimal viewer for inspecting the Lab 4 fixed-base Pupper model.

Run on macOS with:
    mjpython lab4/view_fixed_model.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import mujoco
import mujoco.viewer


_DIR = pathlib.Path(__file__).parent

MODEL_PATH = _DIR / "models" / "pupper_v3_fixed.xml"

def _load_model(path: pathlib.Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = path.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)
    return model, data


def _configure_viewer(viewer: mujoco.viewer.Handle) -> None:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.10]
    viewer.cam.distance = 0.65
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -22


def main() -> int:
    model_path = MODEL_PATH.resolve()
    model, data = _load_model(model_path)

    print(f"Loaded: {model_path}", flush=True)
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}", flush=True)
    print("Opening viewer. Close the window to exit.", flush=True)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            _configure_viewer(viewer)
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
    except RuntimeError as exc:
        if sys.platform == "darwin" and "mjpython" in str(exc):
            print("\nOn macOS, run the interactive viewer with mjpython:", file=sys.stderr)
            print(f"  mjpython {pathlib.Path(__file__)}", file=sys.stderr)
            return 2
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
