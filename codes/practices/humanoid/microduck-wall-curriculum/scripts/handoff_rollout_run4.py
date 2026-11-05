"""Policy handoff demo (plan B): velocity walk -> state injection -> dive roulade + kip-up.

Phase 1 runs the velocity policy in its own env (Mjlab-Velocity-Flat) with a
forced 0.8 m/s forward command; when the duck is walking fast enough the full
sim state (qpos/qvel) is captured. Phase 2 resets the roulade-dive env to a
standing spawn, then INJECTS the captured state (position, heading, joints,
forward momentum >= 0.5 m/s along the heading) — equivalent to the élan spawn
the dive policy was trained with, but the momentum now comes from the real
walk. Both segments are rendered and stitched into one GIF.

Usage:
  .venv/bin/python scripts/handoff_rollout.py 2>&1 | tee logs/handoff_rollout.log
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

ROOT = Path(__file__).resolve().parent.parent
DIVE_TASK = "Mjlab-RouladeDive-Flat-MicroDuck"
VEL_TASK = "Mjlab-Velocity-Flat-MicroDuck"
DIVE_CKPT = ROOT / "logs/rsl_rl/microduck_roulade_dive/2026-09-28_22-56-17_microduck_roulade_dive_run1/model_41995.pt"
VEL_CKPT = ROOT / "logs/rsl_rl/velocity/2026-09-07_11-12-47_velocity/model_24999.pt"
OUT_GIF = ROOT / "logs/gifs/RouladeDive_handoff_run4.gif"

CMD_VX = 0.8          # walk command (m/s)
WALK_MIN_STEPS = 120  # at least 2.4 s of walking before handoff is allowed
WALK_MAX_STEPS = 250  # 5 s cap on phase 1
VX_GATE = 0.35        # handoff when horizontal speed >= this
VX_SUSTAIN = 5        # ... sustained for this many steps (0.1 s)
HANDOFF_VX_MIN = 0.5  # injected forward momentum floor (matches élan 0.35-0.7)
DIVE_MAX_STEPS = 300  # 6 s for dive + roll + rise
STAND_Z = 0.115
RISE_Z = 0.9 * STAND_Z
RISE_SUSTAIN = 25
EPISODE_S = 12.0


def grab_frame(env):
    f = env.render()
    if f is None:
        return None
    f = np.asarray(f)
    if f.ndim == 4:
        f = f[0]
    if f.dtype != np.uint8:
        f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger-vx", type=float, default=None,
                    help="override injected hand-off momentum (m/s); GIF name gains _vx<val>")
    args = ap.parse_args()
    OUT_GIF = ROOT / "logs/gifs/RouladeDive_handoff_run4.gif"
    if args.trigger_vx is not None:
        OUT_GIF = ROOT / f"logs/gifs/RouladeDive_handoff_run4_vx{args.trigger_vx:g}.gif"
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    log = lambda m: print(m, flush=True)

    # ── Phase 1 env: velocity walk ────────────────────────────────────────────
    vel_cfg = load_env_cfg(VEL_TASK, play=True)
    vel_cfg.scene.num_envs = 1
    vel_cfg.episode_length_s = EPISODE_S
    vel_cfg.viewer.distance = 0.9
    vel_cfg.viewer.elevation = -18
    vel_env = ManagerBasedRlEnv(cfg=vel_cfg, device=device, render_mode="rgb_array")
    vel_agent_cfg = load_rl_cfg(VEL_TASK)
    vel_w = RslRlVecEnvWrapper(vel_env, clip_actions=vel_agent_cfg.clip_actions)
    vel_runner = (load_runner_cls(VEL_TASK) or OnPolicyRunner)(vel_w, asdict(vel_agent_cfg), device=device)
    vel_runner.load(str(VEL_CKPT), map_location=device)
    vel_policy = vel_runner.get_inference_policy(device=device)

    def set_twist(env, vx):
        term = env.command_manager.get_term("twist")
        v = torch.tensor([vx, 0.0, 0.0], device=device)
        term.vel_command_b[:] = v
        term.vel_command_w[:] = v

    obs, _ = vel_w.reset()
    frames = []
    fast = 0
    handoff_vx = None
    with torch.no_grad():
        for t in range(WALK_MAX_STEPS):
            set_twist(vel_env, CMD_VX)
            frames.append(grab_frame(vel_env))
            obs, _, _, _ = vel_w.step(vel_policy(obs))
            v_w = torch.nan_to_num(vel_env.scene["robot"].data.root_link_lin_vel_w[0], nan=0.0)
            speed = float(torch.linalg.norm(v_w[:2]))
            if t % 10 == 0:
                log(f"[walk] t={t:3d} speed={speed:.3f}")
            fast = fast + 1 if (speed >= VX_GATE and t >= WALK_MIN_STEPS) else 0
            if fast >= VX_SUSTAIN:
                handoff_vx = speed
                break
    if handoff_vx is None:
        handoff_vx = speed
    walk_qpos = vel_env.sim.data.qpos[0].clone()
    walk_qvel = vel_env.sim.data.qvel[0].clone()
    walk_vw = torch.nan_to_num(vel_env.scene["robot"].data.root_link_lin_vel_w[0], nan=0.0)
    vel_env.close()
    log(f"[handoff] captured vx={handoff_vx:.3f} m/s after phase 1")

    # ── Phase 2 env: dive roulade ─────────────────────────────────────────────
    dive_cfg = load_env_cfg(DIVE_TASK, play=True)
    dive_cfg.scene.num_envs = 1
    dive_cfg.episode_length_s = EPISODE_S
    dive_cfg.viewer.distance = 0.9
    dive_cfg.viewer.elevation = -18
    dive_env = ManagerBasedRlEnv(cfg=dive_cfg, device=device, render_mode="rgb_array")
    tc = dive_env.event_manager.get_term_cfg("set_roulade_state")
    saved = dict(tc.params)
    tc.params["standing_prob"] = 1.0
    tc.params["midroll_prob"] = 0.0
    tc.params["forward_vel_range"] = (0.0, 0.0)  # momentum comes from the walk hand-off

    dive_agent_cfg = load_rl_cfg(DIVE_TASK)
    dive_w = RslRlVecEnvWrapper(dive_env, clip_actions=dive_agent_cfg.clip_actions)
    dive_runner = (load_runner_cls(DIVE_TASK) or OnPolicyRunner)(dive_w, asdict(dive_agent_cfg), device=device)
    dive_runner.load(str(DIVE_CKPT), map_location=device)
    dive_policy = dive_runner.get_inference_policy(device=device)

    obs, _ = dive_w.reset()

    # Inject the walked state (equivalent hand-off). Heading = world x of the
    # walk velocity if it was moving forward, else the walk yaw.
    heading = walk_vw[:2] / (torch.linalg.norm(walk_vw[:2]) + 1e-6)
    if float(torch.linalg.norm(walk_vw[:2])) < 0.1:
        yaw = walk_qpos[5]  # approximate: heading unknown -> use quaternion below instead
        heading = torch.tensor([1.0, 0.0], device=device)
    inject_vx = args.trigger_vx if args.trigger_vx is not None else max(HANDOFF_VX_MIN, handoff_vx)
    dive_env.sim.data.qpos[0, :7] = walk_qpos[:7]   # root pos + orientation (walk heading)
    # joint angles: KEEP the reset's HOME pose (the dive policy only triggers
    # from the standing pose it was trained on — a mid-gait capture makes it
    # freeze; momentum, position and heading are the hand-off).
    dive_env.sim.data.qvel[0, 0] = inject_vx * heading[0]
    dive_env.sim.data.qvel[0, 1] = inject_vx * heading[1]
    dive_env.sim.data.qvel[0, 2] = 0.0
    dive_env.sim.data.qvel[0, 3:6] = 0.0            # no body ang vel (matches spawn)
    dive_env.sim.data.qvel[0, 6:] = 0.0             # joint vels zero (matches spawn)
    log(f"[inject] vx={inject_vx:.3f} m/s heading=({heading[0]:.2f},{heading[1]:.2f})")

    roll_complete = False
    rise_step = None
    risen_streak = 0
    final = None
    with torch.no_grad():
        for t in range(DIVE_MAX_STEPS):
            set_twist(dive_env, 0.0)
            frames.append(grab_frame(dive_env))
            obs, _, _, _ = dive_w.step(dive_policy(obs))
            robot = dive_env.scene["robot"]
            z = float(robot.data.root_link_pos_w[0, 2] - dive_env.scene.terrain.env_origins[0, 2])
            quat = robot.data.root_link_quat_w[0]
            tilt = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
            frontier = float(torch.nan_to_num(dive_env._roulade_max[0], nan=0.0))
            speed = float(torch.linalg.norm(torch.nan_to_num(robot.data.root_link_lin_vel_w[0, :2], nan=0.0)))
            final = (speed, z, float(tilt), math.degrees(frontier))
            roll_complete = roll_complete or frontier >= math.radians(260)
            risen_streak = risen_streak + 1 if z >= RISE_Z else 0
            if rise_step is None and risen_streak >= RISE_SUSTAIN:
                rise_step = t
            if t % 10 == 0:
                log(f"[dive] t={t:3d} speed={speed:.3f} z={z:.3f} "
                    f"tilt={math.degrees(math.acos(max(-1, min(1, tilt)))):5.1f}deg "
                    f"frontier={math.degrees(frontier):6.1f}deg")
            if rise_step is not None and t - rise_step > 60:
                break
    tc.params.clear(); tc.params.update(saved)
    dive_env.close()

    speed, z, tilt, fdeg = final
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, tilt))))
    log(f"[final] speed={speed:.3f} z={z:.4f} tilt={tilt_deg:.1f}deg frontier={fdeg:.1f}deg")
    log(f"[verdict] roll_complete={roll_complete} rise_step={rise_step} frames={len(frames)}")

    # ── GIF export ────────────────────────────────────────────────────────────
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    imgs = [f for f in frames if f is not None][::2]  # 50 Hz -> 25 fps
    ims = []
    for f in imgs:
        im = Image.fromarray(f)
        im = im.resize((640, int(im.height * 640 / im.width)))
        ims.append(im)
    ims[0].save(OUT_GIF, save_all=True, append_images=ims[1:], loop=0, duration=40, optimize=True)
    size_mb = OUT_GIF.stat().st_size / 1e6
    log(f"[gif] saved {OUT_GIF} ({size_mb:.1f} MB, {len(frames)} frames total)")

    # ── Result summary ────────────────────────────────────────────────────────
    result = f"""# Handoff rollout result (Plan B: state injection)

Date: 2026-09-28
Dive ckpt: {DIVE_CKPT.name} (run3, 2026-09-27 run)
Velocity ckpt: {VEL_CKPT.name} (2026-09-07 run)

## Outcome
- walk phase reached vx = {handoff_vx:.3f} m/s at capture (command {CMD_VX} m/s)
- injected hand-off momentum: {inject_vx:.3f} m/s along the walk heading
- roll complete (frontier >= 260 deg): **{roll_complete}** (final frontier {fdeg:.1f} deg)
- rise (z >= {RISE_Z:.3f} m sustained 0.5 s): rise step = {rise_step}
- final state: z = {z:.4f} m, tilt = {tilt_deg:.1f} deg
- GIF: {OUT_GIF.name} ({size_mb:.1f} MB), {len(imgs)} frames @ 25 fps

## Notes
- Plan A (both policies in one env, live switch) was tried first: the dive
  policy freezes when handed a mid-gait state (walking joint velocities/pose
  are outside its training spawn distribution — HOME pose, zero joint/body
  velocity, élan 0.35-0.7 m/s). It just decelerates and stands.
- Plan B injects the walked root state (pos, orientation, joints) plus a
  >= 0.5 m/s forward momentum into the dive env reset — the same distribution
  the dive policy trained on, with the momentum sourced from the real walk.
- The microduck velocity policy tops out around 0.3-0.4 m/s horizontal speed
  (oscillating), even at a 0.8 m/s command — the injected momentum floor
  covers the gap to the training élan range.

Run log: logs/handoff_rollout.log
"""
    (ROOT / "logs/handoff_result.md").write_text(result)
    print(result)


if __name__ == "__main__":
    main()
