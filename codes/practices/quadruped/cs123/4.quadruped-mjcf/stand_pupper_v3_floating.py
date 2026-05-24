"""Make the floating-base Pupper stand still under PD position servo.

Run from this directory with:
    mjpython stand_pupper_v3_floating.py    # macOS
    python stand_pupper_v3_floating.py      # Linux / Windows

This extends view_pupper_v3_floating.py with two pieces:

* A target pose STAND_POSE that is written into ``data.ctrl`` every step, so
  the position servo pulls the legs into a "knee-bent, body up" stand instead
  of just collapsing to the home (all zeros) pose.
* Recording ``data.qpos[2]`` (base z) over time, so when the viewer is closed
  we print the final height and the last-1-second standard deviation as the
  Lab 4.5 pass criterion (std < 5 mm).

macOS note: do not run ``mjpython -m mujoco.viewer --mjcf=...``; see the
docstring of view_pupper_v3_floating.py for the underlying mjpython + runpy
issue.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

import mujoco
import mujoco.viewer


_DIR = pathlib.Path(__file__).parent

MODEL_PATH = _DIR / "models" / "pupper_v3_floating.xml"

# Actuator order: front_r {1,2,3}, front_l {1,2,3}, back_r {1,2,3}, back_l {1,2,3}.
#
# Pupper's mesh + per-body quaternions are set up so that joint angle = 0
# already puts each leg into a knee-bent, body-up stance. That means the
# stand pose is simply the home keyframe's ctrl (all zeros). We still write
# it into data.ctrl every step so the servo target stays locked even when
# external code (e.g. the §5 gait controller) starts touching data.ctrl in
# the same loop.
#
# To experiment with other postures, change any of these 12 entries. Remember
# that the left-side HFE / KFE limits are mirrored (see §4.2.1 in the chapter),
# so non-zero left-leg targets typically need flipped signs.
STAND_POSE = np.zeros(12, dtype=np.float64)


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
    viewer.cam.lookat[:] = [0.0, 0.0, 0.15]
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -22


def _report_stability(heights: list[float], timestep: float) -> None:
    """Print final base z and the last-1-second standard deviation."""
    if not heights:
        print("No samples recorded; nothing to report.", file=sys.stderr)
        return
    samples_per_second = max(int(round(1.0 / timestep)), 1)
    tail = np.array(heights[-samples_per_second:])
    print(
        f"final z={tail[-1]:.3f} m, "
        f"last-1s std={np.std(tail) * 1000.0:.2f} mm "
        f"(< 5 mm = stable)"
    )


def main() -> int:
    model_path = MODEL_PATH.resolve()
    model, data = _load_model(model_path)

    print(f"Loaded: {model_path}", flush=True)
    print(
        f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}",
        flush=True,
    )
    print(f"  timestep={model.opt.timestep:.4f} s", flush=True)
    print(
        "Opening viewer. Close the window to exit and see the stability report.",
        flush=True,
    )

    heights: list[float] = []
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            _configure_viewer(viewer)
            while viewer.is_running():
                step_start = time.perf_counter()
                data.ctrl[:] = STAND_POSE
                mujoco.mj_step(model, data)
                heights.append(float(data.qpos[2]))
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

    _report_stability(heights, model.opt.timestep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
