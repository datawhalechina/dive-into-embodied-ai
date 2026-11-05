"""Record a play rollout video for a trained checkpoint (no ONNX export)."""
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

import mjlab.tasks  # noqa: F401  (populate registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class RolloutConfig:
    checkpoint_file: str
    num_envs: int = 1
    device: str | None = None
    video_length: int = 300
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    distance: float | None = None
    elevation: float | None = None
    azimuth: float | None = None
    speed_overlay: bool = False  # burn live speed/height text onto frames


class _SpeedOverlayWrapper:
    """Phase-A v4 (user 2026-10-14 14:12): GIF 叠加实时速度/高度/腾空状态。

    透明代理 — 只拦截 render()，在帧上烧字 (env0): v=朝向前进速度,
    vz=垂直速度, z=躯干高度, AIR=脚离地。overlay 失败绝不影响 rollout。
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def render(self, *args, **kwargs):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        frame = self._env.render(*args, **kwargs)
        try:
            robot = self._env.scene["robot"]
            vel = robot.data.root_link_vel_w[0].detach().cpu().numpy()
            pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
            rpos = robot.data.root_link_pose_w[0]
            yaw = float(torch.atan2(
                2.0 * rpos[3] * rpos[6], 1.0 - 2.0 * rpos[6] ** 2
            ))
            import math as _m
            fwd = np.array([_m.cos(yaw), _m.sin(yaw)])
            vx = float(vel[:2] @ fwd)
            feet = self._env.scene.sensors.get("feet_ground_contact")
            airborne = (
                feet is not None
                and int((feet.data.found.view(-1) > 0).sum()) == 0
            )
            txt = (f"v={vx:+.2f} m/s  vz={float(vel[2]):+.2f}  "
                   f"z={float(pos[2]):.3f}  {'AIR' if airborne else ''}")
            img = Image.fromarray(frame)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
                )
            except OSError:
                font = ImageFont.load_default()
            d = ImageDraw.Draw(img)
            d.text((26, 28), txt, font=font,
                   stroke_width=4, stroke_fill=(0, 0, 0), fill=(80, 255, 80))
            frame = np.array(img)
        except Exception as e:  # overlay must never kill the rollout
            print(f"[speed_overlay] skip: {e}")
        return frame


def run(task_id: str, cfg: RolloutConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width
    if cfg.camera is not None:
        env_cfg.viewer.camera = cfg.camera
    if cfg.distance is not None:
        env_cfg.viewer.distance = cfg.distance
    if cfg.elevation is not None:
        env_cfg.viewer.elevation = cfg.elevation
    if cfg.azimuth is not None:
        env_cfg.viewer.azimuth = cfg.azimuth

    resume_path = Path(cfg.checkpoint_file)
    log_dir = resume_path.parent
    video_dir = log_dir / "videos" / "play"
    video_dir.mkdir(parents=True, exist_ok=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    if cfg.speed_overlay:
        env = _SpeedOverlayWrapper(env)
    env = VideoRecorder(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=cfg.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(cfg.video_length):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()
    print(f"[video_rollout] done -> {video_dir}")


def main():
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(RolloutConfig, args=remaining_args, prog=sys.argv[0] + f" {chosen_task}")
    run(chosen_task, cfg)


if __name__ == "__main__":
    main()
