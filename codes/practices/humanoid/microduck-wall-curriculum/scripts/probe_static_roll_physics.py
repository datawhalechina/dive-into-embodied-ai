"""Static physics probe: can the Micro Duck body tumble past 180 deg (over the
head) purely from an initial roll angle + angular velocity, zero policy?

Open-loop sweep: init roll 30..300 deg x init roll-rate {0,3,6,10} rad/s,
zero actions, ~300 steps. Records max unwrapped roll, passed-180, final
upright, peak CoM height. CSV -> logs/static_roll_sweep.csv.

APIs used (mjlab, verified in .venv):
  scene["robot"].data.write_root_pose(pose[B,7], env_ids)   # pos+quat(w), world
  scene["robot"].data.write_root_velocity(vel[B,6], env_ids) # lin+ang vel, world frame
  reads: robot.data.root_link_pose_w / root_link_vel_w / root_link_pos_w
"""
import os, sys, math, csv
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("WANDB_MODE", "offline")

import torch
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Roulade-Flat-MicroDuck"
N_ENVS = 1
STEPS = 240  # < episode 5s/0.02 to avoid timeout; break on termination anyway

def quat_from_axis_angle(axis, ang):
    axis = axis / axis.norm()
    half = torch.tensor(float(ang) / 2.0, device=axis.device)
    return torch.cat([torch.cos(half).unsqueeze(0),
                      axis * torch.sin(half).unsqueeze(0)])

def quat_mul(a, b):
    w1, v1 = a[0], a[1:]; w2, v2 = b[0], b[1:]
    return torch.cat([(w1 * w2 - v1 @ v2).unsqueeze(0),
                      w1 * v2 + w2 * v1 + torch.cross(v1, v2)])

def roll_of_quat(q):  # Euler roll (about world x, yaw~0 setup)
    w, x, y, z = q
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.num_envs = N_ENVS
    env_cfg.auto_reset = False   # we manage resets manually; never auto-reset mid-roll
    env = ManagerBasedRlEnv(cfg=env_cfg, device=dev, render_mode="rgb_array")
    env.reset()
    robot = env.scene["robot"]
    data = robot.data
    base_pose = data.root_link_pose_w.clone()   # [B,7] actual post-reset pose
    stand_z = float(base_pose[0, 2])
    print(f"[probe] spawn pose: pos={base_pose[0,:3].tolist()} stand_z={stand_z:.3f}")

    fwd_axis = torch.tensor([1.0, 0.0, 0.0], device=dev)  # forward = +x (yaw 0)

    zero_action = torch.zeros(N_ENVS, env_cfg.actions["joint_pos"].term_cfgs[0].dim
                              if hasattr(env_cfg.actions["joint_pos"], "term_cfgs") else 1,
                              device=dev) if False else None
    # robust zero action: match env action space from a dummy step's doc
    n_act = robot.data.joint_pos_target.shape[-1] if not robot.is_actuated else None
    # fall back: env.step accepts any [N, act_dim]; get from action manager
    n_act = env.action_manager.total_action_dim
    zero_action = torch.zeros(N_ENVS, n_act, device=dev)

    init_rolls = list(range(30, 301, 15))          # deg
    init_ws = [0.0, 3.0, 6.0, 10.0]                # rad/s about forward axis
    rows = []
    for r_deg in init_rolls:
        for w0 in init_ws:
            env.reset()
            ang = math.radians(r_deg)
            q = quat_mul(base_pose[0, 3:7], quat_from_axis_angle(fwd_axis, ang))
            pose = torch.cat([base_pose[0, :3], q]).unsqueeze(0)
            data.write_root_pose(pose, torch.tensor([0], device=dev))
            vel = torch.zeros(1, 6, device=dev)
            vel[0, 3] = w0  # ang_vel world x = forward axis
            data.write_root_velocity(vel, torch.tensor([0], device=dev))

            prev_roll = roll_of_quat(data.root_link_pose_w[0, 3:7].cpu())
            unwrapped = math.radians(r_deg)  # seed with commanded initial roll
            max_roll = unwrapped
            peak_z = float(data.root_link_pos_w[0, 2])
            # NOTE: track from the instant the state is written — the "fall"
            # dynamics after writing IS the experiment. Break on termination.
            for _ in range(STEPS):
                ret = env.step(zero_action)
                terminated = ret.terminated if hasattr(ret, "terminated") else ret[2]
                q_now = data.root_link_pose_w[0, 3:7].cpu()
                r = roll_of_quat(q_now)
                d = r - prev_roll
                while d > math.pi: d -= 2 * math.pi
                while d < -math.pi: d += 2 * math.pi
                unwrapped += d; prev_roll = r
                max_roll = max(max_roll, unwrapped)
                peak_z = max(peak_z, float(data.root_link_pos_w[0, 2]))
                try:
                    done = bool(terminated[0])
                except TypeError:
                    done = bool(terminated)
                if done:
                    break
            final_z = float(data.root_link_pos_w[0, 2])
            final_roll_deg = math.degrees(unwrapped % (2 * math.pi))
            passed_180 = max_roll > math.radians(180) and unwrapped > math.radians(170)
            upright = final_roll_deg < 30 or final_roll_deg > 330
            rows.append(dict(init_roll_deg=r_deg, init_w=w0,
                             max_roll_deg=round(math.degrees(max_roll), 1),
                             passed_180=int(passed_180),
                             final_upright=int(upright and abs(final_z - stand_z) < 0.05),
                             final_roll_deg=round(final_roll_deg, 1),
                             final_z=round(final_z, 3), peak_z=round(peak_z, 3)))
            print(rows[-1])

    out = Path("logs/static_roll_sweep.csv"); out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"\n[probe] wrote {out} ({len(rows)} rows); stand_z={stand_z:.3f}")
    n180 = sum(r["passed_180"] for r in rows)
    nup = sum(r["final_upright"] for r in rows)
    print(f"[probe] passed_180: {n180}/{len(rows)}  final_upright: {nup}/{len(rows)}")

if __name__ == "__main__":
    main()
