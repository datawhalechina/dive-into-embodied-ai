#!/usr/bin/env python3
"""Render a checkpoint with the actual mjlab/MuJoCo-Warp environment.

Unlike the lightweight ``render_checkpoint.py`` deployment-side renderer, this
script keeps the training environment (BAM actuator, observations, command
manager and terminations) intact.  It is useful for deciding whether a
checkpoint is genuinely stable before publishing a GIF.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def _apply_clean_play_config(env_cfg, task: str) -> None:
    """Remove evaluation disturbances without changing the training config.

    ``play=True`` intentionally enables frequent pushes and keeps reset/domain
    randomization so a viewer can stress-test a policy.  That is useful for
    robustness checks but makes a short action-demo video ambiguous: a standup
    clip can start prone, be pushed again before it settles, and then be
    mistaken for a failed policy.  ``--clean`` keeps the real mjlab task and
    reward/observation graph while making the initial condition deterministic.
    """
    env_cfg.curriculum.clear()  # Freeze reset conditions; curricula otherwise overwrite them.
    env_cfg.seed = 42
    events = env_cfg.events

    # No external disturbance during the action demonstration.
    events.pop("push_robot", None)

    # Disable reset-time domain randomization by collapsing each registered
    # event to its nominal value.  Keep the event terms themselves because the
    # training curriculum may still look them up during reset.
    if "foot_friction" in events:
        events["foot_friction"].params["ranges"] = (1.0, 1.0)
    if "encoder_bias" in events:
        events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    if "randomize_com" in events:
        events["randomize_com"].params["ranges"] = (0.0, 0.0)
    if "randomize_head_com" in events:
        events["randomize_head_com"].params["ranges"] = (0.0, 0.0)
    if "randomize_mass_inertia" in events:
        events["randomize_mass_inertia"].params["alpha_range"] = (0.0, 0.0)
    if "randomize_joint_friction" in events:
        events["randomize_joint_friction"].params["scale_range"] = (1.0, 1.0)
    if "randomize_armature" in events:
        events["randomize_armature"].params["ranges"] = (1.0, 1.0)
    if "randomize_motor_gains" in events:
        events["randomize_motor_gains"].params.update(
            kp_range=(1.0, 1.0), kd_range=(1.0, 1.0)
        )
    # The push curriculum references this event, so remove that curriculum
    # entry together with the interval event.
    env_cfg.curriculum.pop("push_magnitude", None)

    task_lower = task.lower()
    if "standup" in task_lower:
        # Demonstration contract: sitting keyframe -> standing.  The policy
        # was trained with the full ground-state curriculum; this only fixes
        # the state sampled at reset for an interpretable MP4.
        term = events.get("set_ground_state")
        if term is not None:
            params = term.params
            params.update(
                face_down_prob=0.0,
                face_up_prob=0.0,
                sitting_prob=1.0,
                standing_prob=0.0,
                sitting_joint_noise_std=0.0,
                sitting_tilt_max=0.0,
                sitting_z_min=0.060,
                sitting_z_max=0.060,
            )
    elif "ballkick" in task_lower:
        # Remove ball placement noise while preserving the trained kick-foot
        # offset and the actual task reset callback.
        term = events.get("reset_ball")
        if term is not None:
            term.params["noise_xy"] = 0.0
    elif "groundpick" in task_lower:
        # GroundPick already starts upright; with pushes and DR removed its
        # phase command is the only changing input in the demo.
        env_cfg.commands["twist"].randomize_phase = False


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
    clean: bool,
) -> dict[str, float]:
    configure_torch_backends()
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.seed = 42
    if clean:
        _apply_clean_play_config(env_cfg, task)
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
    try:
        with torch.inference_mode():
            for _ in range(frames):
                # Keep the visual comparison deterministic.  The policy still
                # receives the complete 61D observation; only command terms are
                # overridden between steps.
                for name in ("head_pose", "body_pose"):
                    if name in env.command_manager.active_terms:
                        env.command_manager.get_command(name).zero_()
                if "groundpick" not in task.lower():
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
    }
    mp4.with_suffix(".json").write_text(json.dumps({"task": task, "checkpoint": str(checkpoint), "seed": 42, "clean": clean, "stats": stats}, indent=2))
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
    parser.add_argument(
        "--clean",
        action="store_true",
        help="fixed action-demo reset with pushes and domain randomization disabled",
    )
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
        args.clean,
    )


if __name__ == "__main__":
    main()
