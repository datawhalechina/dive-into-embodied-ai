"""Locomotion smoothness probe for the VLA CPG executor.

Runs a fixed skill sequence (forward/steer transitions) open-loop and measures
joint-target jerk, body-velocity jerk and vertical bounce. Compares the current
executor against smoothing variants (joint slew-rate limit + steer command ramp).
"""
import os, json, math, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from pathlib import Path

HERE = Path("/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco")
CONFIG = json.loads((HERE / "gait.json").read_text())
DT_PHYS, DT_CTRL, SUB = 0.001, 0.02, 20
SIGNS = np.array(CONFIG["actuator_signs"]); BIAS = np.array(CONFIG["bias_rad"])
AMPS = np.array(CONFIG["amplitude_rad"]); PHASE0 = np.array(CONFIG["phase_rad"])
FREQ = CONFIG["frequency_hz"]; DUTY = CONFIG["sweep_fraction"]; STEER_LIMIT = CONFIG["steering_limit"]
SIDE = {"L": (0, 3), "R": (1, 2)}
RAMP_S = CONFIG["ramp_seconds"]; CTRL_MAX = 1.2
MAX_SLEW = CONFIG.get("max_target_speed_rad_s", 8.0)


def wave_at(ph, duty):
    cycle = (ph / (2 * math.pi)) % 1.0
    phased = np.where(cycle < duty, math.pi * cycle / duty,
                      math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(phased)


def run(sequence, slew=False, steer_ramp=0.0, gif=None):
    model = mujoco.MjModel.from_xml_path(str(HERE / "outputs" / "dataset_w0" / "scene.xml"))
    model.opt.timestep = DT_PHYS
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    renderer = mujoco.Renderer(model, 320, 320) if gif else None
    mujoco.mj_resetData(model, data); data.ctrl[:] = 0; mujoco.mj_forward(model, data)
    for _ in range(int(2.0 / DT_PHYS)):
        mujoco.mj_step(model, data)

    amp_scale, phases_t = 0.0, 0.0
    steer_cmd = 0.0          # smoothed steer command
    target_prev = np.zeros(4)
    ctrl_log, z_log, vel_log, frames = [], [], [], []
    t = 0.0
    DT_FRAME = 0.1           # match eval/collect: ctrl updated at DATA_HZ=10, held 5 ctrl steps
    for skill, dur in sequence:
        steer_target = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
        n_frames = int(dur / DT_FRAME)
        for _ in range(n_frames):
            if steer_ramp > 0:
                ds = DT_FRAME / steer_ramp
                steer_cmd += np.clip(steer_target - steer_cmd, -ds, ds)
            else:
                steer_cmd = steer_target
            amp_scale = min(1.0, amp_scale + DT_FRAME / RAMP_S)
            phases_t += 2 * math.pi * FREQ * DT_FRAME
            duty = np.full(4, DUTY)
            if steer_cmd > 0:
                duty[list(SIDE["L"])] *= (1 - steer_cmd * STEER_LIMIT)
                duty[list(SIDE["R"])] *= (1 + steer_cmd * STEER_LIMIT)
            elif steer_cmd < 0:
                duty[list(SIDE["L"])] *= (1 + abs(steer_cmd) * STEER_LIMIT)
                duty[list(SIDE["R"])] *= (1 - abs(steer_cmd) * STEER_LIMIT)
            w = wave_at(phases_t + PHASE0, duty)
            target = np.clip(SIGNS * (BIAS + AMPS * amp_scale * w), -CTRL_MAX, CTRL_MAX)
            if slew:
                md = MAX_SLEW * DT_FRAME
                target = target_prev + np.clip(target - target_prev, -md, md)
            target_prev = target
            data.ctrl[:] = target
            for _ in range(5):           # CTRL_HZ//DATA_HZ control steps per data frame
                for _ in range(SUB):
                    mujoco.mj_step(model, data)
                t += DT_CTRL
            ctrl_log.append(target.copy())
            z_log.append(float(data.body(body_id).xpos[2]))
            vel_log.append(data.qvel[:3].copy())
            if renderer:
                cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = [0, 0, 0]; cam.distance = 1.0; cam.azimuth = 135; cam.elevation = -30
                renderer.update_scene(data, camera=cam); frames.append(renderer.render().copy())
    ctrl = np.array(ctrl_log); vel = np.array(vel_log); z = np.array(z_log)
    # jerk metrics (2nd difference std), normalized
    ctrl_jerk = float(np.mean(np.std(np.diff(ctrl, n=2, axis=0), axis=0)))
    vel_jerk = float(np.mean(np.std(np.diff(vel, n=2, axis=0), axis=0)))
    bounce = float(np.std(z))
    speed = float(np.mean(np.linalg.norm(vel[:, :2], axis=1)))
    if gif and frames:
        import imageio.v2 as imageio
        imageio.mimsave(gif, frames, fps=25, loop=0)
    return {"ctrl_jerk": ctrl_jerk, "vel_jerk": vel_jerk, "bounce_mm": bounce * 1000,
            "speed_mm_s": speed * 1000}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif-prefix", default=None)
    args = ap.parse_args()
    seq = [("forward", 4), ("steer_L", 3), ("forward", 4), ("steer_R", 3), ("forward", 4)]
    variants = {
        "baseline": dict(slew=False, steer_ramp=0.0),
        "slew": dict(slew=True, steer_ramp=0.0),
        "slew+ramp0.3": dict(slew=True, steer_ramp=0.3),
        "slew+ramp0.5": dict(slew=True, steer_ramp=0.5),
    }
    for name, kw in variants.items():
        gif = f"{args.gif_prefix}_{name}.gif" if args.gif_prefix else None
        m = run(seq, gif=gif, **kw)
        print(f"{name:16s} ctrl_jerk={m['ctrl_jerk']:.5f} vel_jerk={m['vel_jerk']:.5f} "
              f"bounce={m['bounce_mm']:.2f}mm speed={m['speed_mm_s']:.1f}mm/s", flush=True)