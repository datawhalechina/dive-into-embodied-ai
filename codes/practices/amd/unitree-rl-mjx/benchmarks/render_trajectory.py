"""Render a saved qpos trajectory to MP4, CPU-only.

Trajectories come from the training entries; rendering happens wherever a
working GL stack is, so the GPU boxes never need one.

    uv run python benchmarks/render_trajectory.py runs/r0/trajectory.npz --out go1.mp4
    uv run python benchmarks/render_trajectory.py --env go2 runs/g0/trajectory.npz --out go2.mp4
"""

import argparse

import mediapy
import mujoco
import numpy as np
from mujoco_playground import registry

ENV_NAME = "Go1JoystickFlatTerrain"
ENV_OVERRIDES = {"impl": "jax"}
FPS = 50  # Both envs' control rate (ctrl_dt 0.02).
TRACK_BODY = {"go1-playground": "trunk", "go2": "base_link"}


def _load_model(env_name: str) -> mujoco.MjModel:
  if env_name == "go2":
    # The standalone scene: same kinematics, plus floor, lights and materials.
    from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2

    return go2.get_go2_scene_spec().compile()
  return registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES).mj_model


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("trajectory", help="trajectory.npz with a qpos array")
  parser.add_argument("--out", required=True)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument(
    "--azimuth",
    type=float,
    default=None,
    help="world-frame camera angle; set it perpendicular to the direction of "
    "travel for a side view. Without it the model's own track camera is used.",
  )
  parser.add_argument("--distance", type=float, default=2.2)
  parser.add_argument("--elevation", type=float, default=-12.0)
  parser.add_argument(
    "--env", choices=("go1-playground", "go2"), default="go1-playground"
  )
  args = parser.parse_args()

  qpos = np.load(args.trajectory)["qpos"]
  model = _load_model(args.env)
  data = mujoco.MjData(model)

  if args.azimuth is None:
    camera = "track" if model.ncam else -1
  else:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY[args.env]
    )
    camera.distance, camera.azimuth = args.distance, args.azimuth
    camera.elevation = args.elevation

  frames = []
  with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
    for q in qpos:
      data.qpos[:] = q
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera=camera)
      frames.append(renderer.render())

  mediapy.write_video(args.out, frames, fps=FPS)
  print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
  main()
