#!/usr/bin/env python3
"""Render a MicroDuck ONNX checkpoint without opening a viewer window.

This is deliberately a small, CPU-side playback tool.  It uses the same
``PolicyInference`` observation implementation as ``infer_policy.py`` and
MuJoCo's off-screen renderer, so it works on headless development machines
while a GPU training job is running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from infer_policy import DEFAULT_POSE, MICRODUCK_XML, PolicyInference


def _make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.10)
    camera.distance = 0.62
    camera.azimuth = 100.0
    camera.elevation = -8.0
    return camera


def render(
    onnx: Path,
    mp4: Path,
    gif: Path | None,
    frames: int,
    width: int,
    height: int,
    lin_vel_x: float,
    lin_vel_y: float,
    ang_vel_z: float,
) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(Path(MICRODUCK_XML)))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    freejoint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    data.qpos[qpos_adr : qpos_adr + 3] = (0.0, 0.0, 0.125)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    for joint_id, actuator_id in enumerate(range(model.nu)):
        joint_qpos = int(model.jnt_qposadr[model.actuator_trnid[actuator_id, 0]])
        data.qpos[joint_qpos] = DEFAULT_POSE[joint_id]
        data.ctrl[actuator_id] = DEFAULT_POSE[joint_id]
    mujoco.mj_forward(model, data)

    # The velocity task in this tutorial uses the unified 13D command slots
    # (61D actor observation).  Keep this explicit so a 51D legacy ONNX cannot
    # silently be paired with the wrong observation layout.
    policy = PolicyInference(
        model, data, walking_onnx_path=str(onnx), new_cmd_obs=True
    )
    policy.vel_max_x = 0.3
    policy.vel_min_x = -0.3
    policy.vel_max_y = 0.2
    policy.vel_min_y = -0.2
    policy.vel_max_ang = 1.5
    policy.set_vel_cmd(lin_vel_x, lin_vel_y, ang_vel_z)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _make_camera()
    mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(mp4, fps=50, codec="libx264", quality=8)
    gif_writer = None
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
        gif_writer = imageio.get_writer(gif, mode="I", duration=0.05, loop=0)

    min_z = float("inf")
    min_upright = float("inf")
    fallen_frames = 0
    try:
        for _ in range(frames):
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(4):
                mujoco.mj_step(model, data)

            trunk_z = float(data.qpos[qpos_adr + 2])
            quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]
            # For [w, x, y, z], the body-frame gravity z component is a cheap
            # upright proxy: -1 is upright and values near 0 mean horizontal.
            upright = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
            min_z = min(min_z, trunk_z)
            min_upright = min(min_upright, upright)
            if trunk_z < 0.08 or upright < 0.35:
                fallen_frames += 1

            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            writer.append_data(frame)
            if gif_writer is not None:
                gif_writer.append_data(frame)
    finally:
        writer.close()
        if gif_writer is not None:
            gif_writer.close()
        renderer.close()

    stats = {
        "frames": float(frames),
        "duration_s": frames / 50.0,
        "min_trunk_z_m": min_z,
        "min_upright_proxy": min_upright,
        "fallen_frames": float(fallen_frames),
        "fallen_fraction": fallen_frames / max(frames, 1),
    }
    print(f"saved {mp4}")
    if gif is not None:
        print(f"saved {gif}")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--lin-vel-x", type=float, default=0.15)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    render(
        args.onnx,
        args.mp4,
        args.gif,
        args.frames,
        args.width,
        args.height,
        args.lin_vel_x,
        args.lin_vel_y,
        args.ang_vel_z,
    )


if __name__ == "__main__":
    main()
