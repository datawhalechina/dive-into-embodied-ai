"""Inspect the Lab 4 floating-base Pupper model in MuJoCo viewer.

Run from this directory with:
    mjpython view_pupper_v3_floating.py    # macOS
    python view_pupper_v3_floating.py      # Linux / Windows

Unlike view_pupper_v3_fixed.py, the main loop here calls mj_step instead of
mj_forward, so the simulator actually evolves gravity and contacts. You should
see the robot fall from z=0.28 m, the feet touch the ground, and the position
servo pull the 12 leg joints back toward their home targets.

macOS note: do not run ``mjpython -m mujoco.viewer --mjcf=...`` — mjpython
already imports ``mujoco.viewer`` during startup (to claim the GUI main
thread), so re-executing it via ``-m``/runpy raises
``RuntimeError: Caught an unknown exception!`` at ``_Simulate(...)``. Use the
script entry (``mjpython view_pupper_v3_floating.py``) to bypass this.
"""

from __future__ import annotations

import pathlib
import sys
import time

import mujoco
import mujoco.viewer


_DIR = pathlib.Path(__file__).parent

MODEL_PATH = _DIR / "models" / "pupper_v3_floating.xml"


def _load_model(path: pathlib.Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the MJCF model and put it into the home keyframe if available."""
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
    """Frame the falling robot from a 3/4 angle so the drop is easy to see."""
    viewer.cam.lookat[:] = [0.0, 0.0, 0.15]
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -22


def main() -> int:
    model_path = MODEL_PATH.resolve()
    model, data = _load_model(model_path)

    print(f"Loaded: {model_path}", flush=True)
    print(
        f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}",
        flush=True,
    )
    print(f"  timestep={model.opt.timestep:.4f} s", flush=True)
    print("Opening viewer. Close the window to exit.", flush=True)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            _configure_viewer(viewer)
            while viewer.is_running():
                step_start = time.perf_counter()
                mujoco.mj_step(model, data)
                viewer.sync()

                elapsed = time.perf_counter() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    except RuntimeError as exc:
        if sys.platform == "darwin" and "mjpython" in str(exc):
            print("\nOn macOS, run the interactive viewer with mjpython:", file=sys.stderr)
            print(f"  mjpython {pathlib.Path(__file__)}", file=sys.stderr)
            return 2
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
