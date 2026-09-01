#!/usr/bin/env python3
"""Render a checkpoint with the actual mjlab/MuJoCo-Warp environment.

Unlike the lightweight ``render_checkpoint.py`` deployment-side renderer, this
script keeps the training environment (BAM actuator, observations, command
manager and terminations) intact.  It is useful for deciding whether a
checkpoint is genuinely stable before publishing a GIF.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def render(
    task: str,
    checkpoint: Path,
    mp4: Path,
    gif: Path | None,
    frames: int,
    width: int,
    height: int,
    distance: float,
    device: str,
    lin_vel_x: float,
    lin_vel_y: float,
    ang_vel_z: float,
    seed: int,
) -> dict[str, float]:
    configure_torch_backends()
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    env_cfg.viewer.distance = distance
    agent_cfg = load_rl_cfg(task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task)
    if runner_cls is None:
        raise RuntimeError(f"no runner registered for {task}")
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    obs, _ = wrapped.reset()

    mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(mp4, fps=50, codec="libx264", quality=8)
    gif_writer = None
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
        gif_writer = imageio.get_writer(gif, mode="I", duration=0.05, loop=0)

    robot = env.scene["robot"]
    min_z = float("inf")
    min_upright = float("inf")
    done_count = 0
    fell_like_frames = 0
    start_x = float(robot.data.root_link_pos_w[0, 0].item())
    final_x = start_x
    sum_vel_x = 0.0
    sum_abs_vel_error_x = 0.0
    try:
        with torch.inference_mode():
            for _ in range(frames):
                # Keep the visual comparison deterministic.  The policy still
                # receives the complete 61D observation; only command terms are
                # overridden between steps.
                for name in ("head_pose", "body_pose"):
                    if name in env.command_manager.active_terms:
                        env.command_manager.get_command(name).zero_()
                twist = env.command_manager.get_command("twist")
                twist.zero_()
                twist[0, 0] = lin_vel_x
                twist[0, 1] = lin_vel_y
                twist[0, 2] = ang_vel_z

                action = policy(obs)
                obs, _reward, dones, _extras = wrapped.step(action)
                done_count += int(dones[0].item())

                pos = robot.data.root_link_pos_w[0]
                quat = robot.data.root_link_quat_w[0]
                trunk_z = float(pos[2].item())
                final_x = float(pos[0].item())
                vel_x = float(robot.data.root_link_lin_vel_w[0, 0].item())
                sum_vel_x += vel_x
                sum_abs_vel_error_x += abs(vel_x - lin_vel_x)
                # Quaternion convention is [w, x, y, z].  This is the world
                # gravity z component expressed in the body frame.
                upright = float(1.0 - 2.0 * (quat[1].item() ** 2 + quat[2].item() ** 2))
                min_z = min(min_z, trunk_z)
                min_upright = min(min_upright, upright)
                if trunk_z < 0.08 or upright < 0.35:
                    fell_like_frames += 1

                frame = env.render()
                if frame is None:
                    raise RuntimeError("mjlab did not return an RGB frame")
                writer.append_data(frame)
                if gif_writer is not None:
                    gif_writer.append_data(frame)
    finally:
        writer.close()
        if gif_writer is not None:
            gif_writer.close()
        wrapped.close()

    stats = {
        "frames": float(frames),
        "duration_s": frames / 50.0,
        "done_count": float(done_count),
        "min_trunk_z_m": min_z,
        "min_upright_proxy": min_upright,
        "fell_like_frames": float(fell_like_frames),
        "fell_like_fraction": fell_like_frames / max(frames, 1),
        "forward_displacement_m": final_x - start_x,
        "mean_forward_velocity_m_s": sum_vel_x / max(frames, 1),
        "mean_abs_velocity_error_m_s": sum_abs_vel_error_x / max(frames, 1),
    }
    print(f"saved {mp4}")
    if gif is not None:
        print(f"saved {gif}")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--distance", type=float, default=0.8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lin-vel-x", type=float, default=0.15)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    render(
        args.task,
        args.checkpoint,
        args.mp4,
        args.gif,
        args.frames,
        args.width,
        args.height,
        args.distance,
        args.device,
        args.lin_vel_x,
        args.lin_vel_y,
        args.ang_vel_z,
        args.seed,
    )


if __name__ == "__main__":
    main()
