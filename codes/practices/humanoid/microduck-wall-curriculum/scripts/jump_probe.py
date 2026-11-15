"""Physics probe: scripted max jump for the dive-roll task (AGENTS.md: verify
physics assumptions in sim BEFORE training).

Squat -> explosive leg extension -> measure apex height, airtime, liftoff vz.
These numbers set the dive-roll's forward_vel range and airborne reward window.

Usage: .venv/bin/python scripts/jump_probe.py [--squat 0.8] [--extend 1.2]
"""
import argparse

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Roulade-Flat-MicroDuck"

# Servo indices: 0-4 left (hip_yaw, hip_roll, hip_pitch, knee, ankle), 9-13 right.
# Squat (flexion) signs from TUCK_OVERRIDES: L hip-, knee+, ankle+; R mirrored.
LEFT = dict(hip_pitch=2, knee=3, ankle=4)
RIGHT = dict(hip_pitch=11, knee=12, ankle=13)


def leg_action(depth: float, device) -> torch.Tensor:
    """Raw action (offset from HOME) for a symmetric squat of the given depth.

    depth > 0 = squat (flexion), depth < 0 = extension (jump push-off).
    """
    a = torch.zeros(14, device=device)
    a[LEFT["hip_pitch"]] = -depth
    a[LEFT["knee"]] = depth
    a[LEFT["ankle"]] = depth
    a[RIGHT["hip_pitch"]] = depth
    a[RIGHT["knee"]] = -depth
    a[RIGHT["ankle"]] = -depth
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squat", type=float, default=0.8)
    ap.add_argument("--extend", type=float, default=1.2)
    ap.add_argument("--num-envs", type=int, default=64)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Force plain standing spawns via the event manager (never env.cfg).
    tc = env.event_manager.get_term_cfg("set_roulade_state")
    tc.params["standing_prob"] = 1.0
    tc.params["midroll_prob"] = 0.0
    obs, _ = env.reset()

    robot = env.scene["robot"]
    contact = env.scene.sensors["feet_ground_contact"]

    HORIZON = 250  # 5 s
    T_HOLD, T_SQUAT, T_JUMP = 20, 45, 45  # stand 0.4s, squat 0.5s, push from t=45
    z_hist, vz_hist, air_hist = [], [], []

    with torch.no_grad():
        for t in range(HORIZON):
            z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
            vz = robot.data.root_link_lin_vel_w[:, 2]
            found = contact.data.found
            feet = (found.view(found.shape[0], -1) > 0).any(dim=-1)
            z_hist.append(z.cpu()); vz_hist.append(vz.cpu()); air_hist.append((~feet).cpu())

            if t < T_HOLD:
                a = leg_action(0.0, device)
            elif t < T_JUMP:
                a = leg_action(args.squat, device)
            else:
                a = leg_action(-args.extend, device)
            obs, _, _, _, _ = env.step(a.unsqueeze(0).expand(args.num_envs, -1))

    z = torch.stack(z_hist)      # [T, N]
    vz = torch.stack(vz_hist)
    air = torch.stack(air_hist)  # bool [T, N]

    N = z.shape[1]
    apex, liftoff_vz, airtime_s = [], [], []
    for i in range(N):
        air_idx = torch.nonzero(air[:, i]).flatten()
        if len(air_idx) == 0:
            continue
        idx = air_idx.tolist()
        # longest contiguous airborne run after the push
        runs, cur = [], [idx[0]]
        for a, b in zip(idx, idx[1:]):
            if b == a + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        run = max(runs, key=len)
        t0, t1 = run[0], run[-1]
        airtime_s.append((t1 - t0 + 1) * 0.02)
        liftoff_vz.append(float(vz[t0, i]))
        apex.append(float(z[t0 : t1 + 1, i].max()))

    def pct(v, q):
        s = sorted(v)
        return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")

    print(f"squat={args.squat} extend={args.extend}  envs={N}  airborne={len(apex)}/{N}")
    if apex:
        z_stand = float(z[:T_HOLD].median())
        print(f"  stand z          : {z_stand:.4f}")
        print(f"  apex z    p50/p90: {pct(apex, 0.5):.4f} / {pct(apex, 0.9):.4f}")
        print(f"  apex gain p50/p90: {pct(apex, 0.5) - z_stand:+.4f} / {pct(apex, 0.9) - z_stand:+.4f}  (over standing)")
        print(f"  liftoff vz p50/p90: {pct(liftoff_vz, 0.5):.3f} / {pct(liftoff_vz, 0.9):.3f} m/s")
        print(f"  airtime   p50/p90: {pct(airtime_s, 0.5):.3f} / {pct(airtime_s, 0.9):.3f} s")
        # rotation budget: how much of a flip fits in the airtime
        t = pct(airtime_s, 0.5)
        for w in (5.5, 8.0, 12.0):
            print(f"  @ {w:4.1f} rad/s -> {w * t:5.2f} rad = {w * t * 57.3:5.0f} deg in p50 airtime")


if __name__ == "__main__":
    main()
