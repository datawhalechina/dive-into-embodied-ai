"""加载 PPO checkpoint，渲染命令演示和速度跟踪曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from pupper_env import PupperEnv  # noqa: E402
from train import require_rocm_device  # noqa: E402


LAB_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = LAB_DIR / "outputs"
CMD_SCRIPT = [
    ((0.0, 0.0, 0.0), 2.0),
    ((0.5, 0.0, 0.0), 3.0),
    ((0.0, 0.0, 0.5), 2.5),
    ((-0.3, 0.0, 0.0), 2.5),
    ((0.0, 0.0, 0.0), 2.0),
]
GIF_FPS = 12
FRAME_W, FRAME_H = 640, 480


def _cmd_at(time_s: float):
    elapsed = 0.0
    for command, duration in CMD_SCRIPT:
        if time_s < elapsed + duration:
            return command
        elapsed += duration
    return CMD_SCRIPT[-1][0]


def render_demo(model, env, out_path: Path) -> None:
    total_seconds = sum(duration for _, duration in CMD_SCRIPT)
    n_steps = int(total_seconds / env.dt)
    renderer = mujoco.Renderer(env.model, height=FRAME_H, width=FRAME_W)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, camera)
    camera.distance = 1.4
    camera.elevation = -20.0
    camera.azimuth = 90.0

    obs, _ = env.reset(seed=42)
    frames = []
    next_frame_time = 0.0
    try:
        for step_index in range(n_steps):
            time_s = step_index * env.dt
            env.cmd = np.array(_cmd_at(time_s), dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            if time_s + 0.5 * env.dt >= next_frame_time:
                camera.lookat[:] = env.data.xpos[env._base_id]
                renderer.update_scene(env.data, camera=camera)
                frames.append(renderer.render())
                next_frame_time += 1.0 / GIF_FPS

            if terminated or truncated:
                obs, _ = env.reset(seed=42)
    finally:
        renderer.close()

    imageio.mimsave(out_path, frames, duration=1000 / GIF_FPS, loop=0)


def render_velocity(
    model,
    env,
    out_path: Path,
    vx_commands=(0.2, 0.4, 0.6),
    seconds=6.0,
) -> None:
    figure, axes = plt.subplots(
        1, len(vx_commands), figsize=(5 * len(vx_commands), 4), sharey=True,
    )
    if len(vx_commands) == 1:
        axes = [axes]

    n_steps = int(seconds / env.dt)
    for axis, vx_command in zip(axes, vx_commands):
        obs, _ = env.reset(seed=42)
        env.cmd = np.array([vx_command, 0.0, 0.0], dtype=np.float32)
        timestamps = []
        actual_velocities = []

        for step_index in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            timestamps.append(step_index * env.dt)
            rotation = env.data.xmat[env._base_id].reshape(3, 3)
            actual_velocities.append(float(
                (rotation.T @ env.data.qvel[0:3])[0],
            ))

            if terminated or truncated:
                obs, _ = env.reset(seed=42)
                env.cmd = np.array([vx_command, 0.0, 0.0], dtype=np.float32)

        axis.axhline(
            vx_command,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"cmd = {vx_command}",
        )
        axis.plot(
            timestamps,
            actual_velocities,
            color="tab:blue",
            linewidth=1.0,
            label="actual vx",
        )
        axis.set_xlabel("t [s]")
        axis.set_title(f"vx_cmd = {vx_command} m/s")
        axis.legend(loc="lower right", fontsize=9)
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("vx [m/s]")
    figure.suptitle("Velocity tracking")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Pupper PPO checkpoint")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_OUTPUT_DIR / "pupper_ppo.zip"),
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PupperEnv()
    model = PPO.load(
        args.checkpoint,
        env=env,
        device=require_rocm_device(),
    )
    render_demo(model, env, output_dir / "demo.gif")
    render_velocity(model, env, output_dir / "velocity_tracking.png")
    print(
        f"已生成 {output_dir / 'demo.gif'} 和 "
        f"{output_dir / 'velocity_tracking.png'}",
    )


if __name__ == "__main__":
    main()
