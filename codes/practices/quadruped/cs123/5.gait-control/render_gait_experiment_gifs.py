from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[5]
EXERCISES_DIR = ROOT / "codes/practices/quadruped/cs123/exercises"
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from shared.kinematics.leg_kinematics import HIP_OFFSETS, LEG_ORDER, ik_pupper_leg  # noqa: E402


FIG_DIR = ROOT / "docs/practices/quadruped/cs123/figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_GAIT_DEMO = Path(__file__).with_name("pupper_gait_demo.xml")

JOINT_SUFFIXES = ("HAA", "HFE", "KFE")
PHASE_OFFSETS = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}

T_CYCLE = 0.4
DUTY = 0.5
STEP_HEIGHT = 0.04
STAND_HEIGHT = 0.18
MAX_TORQUE = 18.0

WIDTH = 720
HEIGHT = 540
FPS = 24
SETTLE_SECONDS = 0.35
RENDER_SECONDS = 3.2

def leg_phase(t: float, leg: str, t_cycle: float, duty: float = DUTY) -> tuple[bool, float]:
    t_global = (t / t_cycle) % 1.0
    t_local = (t_global + PHASE_OFFSETS[leg]) % 1.0
    if t_local < duty:
        return True, t_local / duty
    return False, (t_local - duty) / (1.0 - duty)


def foot_trajectory(s: float, in_stance: bool, step_length: float, step_height: float, stand_height: float) -> np.ndarray:
    if in_stance:
        x = step_length * (0.5 - s)
        z = -stand_height
    else:
        x = step_length * (s - 0.5)
        z = -stand_height + step_height * np.sin(np.pi * s)
    return np.array((x, 0.0, z), dtype=float)


def make_q_seed() -> dict[str, np.ndarray]:
    return {leg: np.array((0.0, 0.18, -0.36), dtype=float) for leg in LEG_ORDER}


def gait_step(
    t: float,
    step_length: float,
    q_seed: dict[str, np.ndarray],
    *,
    t_cycle: float,
    step_height: float,
    stand_height: float,
) -> np.ndarray:
    target_q = np.zeros(12, dtype=float)
    for k, leg in enumerate(LEG_ORDER):
        in_stance, s = leg_phase(t, leg, t_cycle)
        hip_local = foot_trajectory(s, in_stance, step_length, step_height, stand_height)
        foot_xyz = HIP_OFFSETS[leg] + hip_local
        q_leg = ik_pupper_leg(foot_xyz, leg=leg, q_seed=q_seed[leg])
        q_seed[leg] = q_leg
        target_q[3 * k : 3 * k + 3] = q_leg
    return target_q


def joint_names() -> tuple[str, ...]:
    return tuple(f"{leg}_{suffix}" for leg in LEG_ORDER for suffix in JOINT_SUFFIXES)


def actuator_names() -> tuple[str, ...]:
    return tuple(f"{leg}_{suffix}_motor" for leg in LEG_ORDER for suffix in JOINT_SUFFIXES)


def joint_qpos_qvel_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_ids: list[int] = []
    qvel_ids: list[int] = []
    for name in joint_names():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing joint {name!r} in {model.names!r}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_ids, dtype=int), np.asarray(qvel_ids, dtype=int)


def actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for name in actuator_names():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"missing actuator {name!r} in {model.names!r}")
        ids.append(int(actuator_id))
    return np.asarray(ids, dtype=int)


def body_id(model: mujoco.MjModel, name: str = "base") -> int:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise ValueError(f"missing body {name!r}")
    return int(body)


def reset_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_ids: np.ndarray,
    q0: np.ndarray,
    *,
    stand_height: float,
    weld_active: bool,
) -> None:
    mujoco.mj_resetData(model, data)
    if model.neq:
        data.eq_active[:] = 1 if weld_active else 0
    data.qpos[:7] = (0.0, 0.0, stand_height, 1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_ids] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def apply_pd(
    data: mujoco.MjData,
    qpos_ids: np.ndarray,
    qvel_ids: np.ndarray,
    ctrl_ids: np.ndarray,
    target_q: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> None:
    q = data.qpos[qpos_ids]
    dq = data.qvel[qvel_ids]
    tau = kp * (target_q - q) - kd * dq
    data.ctrl[ctrl_ids] = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)


def add_label(frame: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((18, 18, 438, 68), radius=10, fill=(255, 255, 255, 218))
    draw.text((34, 34), text, fill=(20, 30, 40, 255))
    return image


def make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.82
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.lookat[:] = (0.0, 0.0, 0.10)
    return camera


def render_experiment(
    name: str,
    model_path: Path,
    output: Path,
    *,
    step_length: float,
    t_cycle: float,
    step_height: float,
    stand_height: float,
    kp_value: float,
    kd_value: float,
    weld_active: bool,
    weld_speed: float = 0.0,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    data = mujoco.MjData(model)
    qpos_ids, qvel_ids = joint_qpos_qvel_ids(model)
    ctrl_ids = actuator_ids(model)
    base_id = body_id(model)
    kp = np.full(12, kp_value)
    kd = np.full(12, kd_value)

    q_seed = make_q_seed()
    q0 = gait_step(
        0.0,
        step_length,
        q_seed,
        t_cycle=t_cycle,
        step_height=step_height,
        stand_height=stand_height,
    )
    reset_pose(model, data, qpos_ids, q0, stand_height=stand_height, weld_active=weld_active)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = make_camera()
    frames: list[Image.Image] = []
    next_frame_time = 0.0

    try:
        while data.time < SETTLE_SECONDS + RENDER_SECONDS:
            gait_t = max(0.0, data.time - SETTLE_SECONDS)
            if weld_active and model.neq:
                model.eq_data[0, 0] = weld_speed * gait_t
            target_q = (
                q0
                if data.time < SETTLE_SECONDS
                else gait_step(
                    gait_t,
                    step_length,
                    q_seed,
                    t_cycle=t_cycle,
                    step_height=step_height,
                    stand_height=stand_height,
                )
            )
            apply_pd(data, qpos_ids, qvel_ids, ctrl_ids, target_q, kp, kd)
            mujoco.mj_step(model, data)

            if data.time < SETTLE_SECONDS:
                continue

            render_t = data.time - SETTLE_SECONDS
            if render_t + 0.5 * model.opt.timestep < next_frame_time:
                continue

            camera.lookat[:] = data.xpos[base_id]
            camera.lookat[2] = max(float(camera.lookat[2]), 0.09)
            renderer.update_scene(data, camera=camera)
            frames.append(add_label(renderer.render(), name))
            next_frame_time += 1.0 / FPS
    finally:
        renderer.close()

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=True,
    )
    print(f"saved {output} ({len(frames)} frames, xml={model_path}, weld_active={weld_active})")


def main() -> None:
    render_experiment(
        name="In-place trot · pupper_gait_demo.xml",
        model_path=MODEL_GAIT_DEMO,
        output=FIG_DIR / "lab5_inplace_trot.gif",
        step_length=0.0,
        t_cycle=T_CYCLE,
        step_height=STEP_HEIGHT,
        stand_height=STAND_HEIGHT,
        kp_value=30.0,
        kd_value=1.0,
        weld_active=True,
    )
    render_experiment(
        name="Forward trot · pupper_gait_demo.xml",
        model_path=MODEL_GAIT_DEMO,
        output=FIG_DIR / "lab5_forward_trot.gif",
        step_length=0.07,
        t_cycle=0.8,
        step_height=0.05,
        stand_height=0.16,
        kp_value=24.0,
        kd_value=0.8,
        weld_active=True,
        weld_speed=0.12,
    )


if __name__ == "__main__":
    main()
