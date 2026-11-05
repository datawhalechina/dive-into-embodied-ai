"""Batch eval of the walk->dive hand-off across trigger speeds.

For each vx in --vx-list, runs (--batches x --envs) rollouts (each batch a
fresh seeded env): walk with the velocity policy until each env sustains the
walk-speed gate, snapshot root state + heading, then inject into the dive env
(standing HOME pose) with momentum exactly = trigger vx along that heading.
Reports roll completion, full-2pi rate, kip-up success, over-rotation, and
landing pose. The dive policy trained on élan 0.35-0.7 m/s — higher vx is
deliberate OOD probing.

Usage:
  .venv/bin/python scripts/handoff_eval_vx.py 2>&1 | tee logs/handoff_eval_vx.log
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

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
DIVE_CKPT = ROOT / "logs/rsl_rl/microduck_roulade_dive/2026-09-27_21-31-04_microduck_roulade_dive_run1/model_31996.pt"
VEL_CKPT = ROOT / "logs/rsl_rl/velocity/2026-09-07_11-12-47_velocity/model_24999.pt"
OUT_JSON = ROOT / "logs/handoff_eval_vx.json"

STAND_Z = 0.115
RISE_Z = 0.9 * STAND_Z
RISE_SUSTAIN = 25
WALK_CMD = 0.8
WALK_GATE = 0.35
WALK_SUSTAIN = 5
WALK_MIN_STEPS = 120
WALK_MAX_STEPS = 300
DIVE_STEPS = 250
EPISODE_S = 14.0


def build(task, ckpt, num_envs, device):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = EPISODE_S
    base = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = load_rl_cfg(task)
    w = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task))(w, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), map_location=device)
    return base, w, runner.get_inference_policy(device=device)


def read_state(base, frontier=False):
    robot = base.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2] - base.scene.terrain.env_origins[:, 2]
    quat = robot.data.root_link_quat_w
    tilt_up = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    speed = torch.linalg.norm(
        torch.nan_to_num(robot.data.root_link_lin_vel_w[:, :2], nan=0.0), dim=1
    )
    fr = torch.nan_to_num(base._roulade_max.clone(), nan=0.0) if frontier else None
    return z, tilt_up, speed, fr


def run_batch(trigger_vx, seed, num_envs, device):
    torch.manual_seed(seed)
    # ── walk phase ────────────────────────────────────────────────────────────
    vbase, vw, vpol = build(VEL_TASK, VEL_CKPT, num_envs, device)
    obs, _ = vw.reset()
    n = num_envs
    captured = torch.zeros(n, dtype=torch.bool, device=device)
    snap_qpos = torch.zeros(n, 7, device=device)
    snap_heading = torch.zeros(n, 2, device=device)
    streak = torch.zeros(n, device=device)
    with torch.no_grad():
        for t in range(WALK_MAX_STEPS):
            term = vbase.command_manager.get_term("twist")
            v = torch.tensor([WALK_CMD, 0.0, 0.0], device=device)
            term.vel_command_b[:] = v
            term.vel_command_w[:] = v
            obs, _, _, _ = vw.step(vpol(obs))
            _, _, speed, _ = read_state(vbase, frontier=False)
            vel_w = torch.nan_to_num(vbase.scene["robot"].data.root_link_lin_vel_w[:, :2], nan=0.0)
            gate_ok = (speed >= WALK_GATE) & (t >= WALK_MIN_STEPS)
            streak = torch.where(gate_ok, streak + 1, torch.zeros_like(streak))
            newly = (~captured) & (streak >= WALK_SUSTAIN)
            if newly.any():
                qpos = vbase.sim.data.qpos[newly, :7].clone()
                hv = vel_w[newly]
                hn = torch.linalg.norm(hv, dim=1, keepdim=True).clamp_min(1e-6)
                snap_qpos[newly] = qpos
                snap_heading[newly] = hv / hn
                captured |= newly
            if bool(captured.all()):
                break
    # uncaptured envs: use final state, heading = +x fallback
    if bool((~captured).any()):
        rem = ~captured
        snap_qpos[rem] = vbase.sim.data.qpos[rem, :7].clone()
        snap_heading[rem] = torch.tensor([1.0, 0.0], device=device).expand(int(rem.sum()), 2)
    vbase.close()

    # ── dive phase ────────────────────────────────────────────────────────────
    torch.manual_seed(seed + 1)
    dbase, dw, dpol = build(DIVE_TASK, DIVE_CKPT, n, device)
    tc = dbase.event_manager.get_term_cfg("set_roulade_state")
    tc.params["standing_prob"] = 1.0
    tc.params["midroll_prob"] = 0.0
    tc.params["forward_vel_range"] = (0.0, 0.0)
    obs, _ = dw.reset()
    dbase.sim.data.qpos[:, :7] = snap_qpos  # root pos + walk heading; joints stay HOME
    dbase.sim.data.qvel[:, :6] = 0.0
    dbase.sim.data.qvel[:, 6:] = 0.0
    dbase.sim.data.qvel[:, 0] = trigger_vx * snap_heading[:, 0]
    dbase.sim.data.qvel[:, 1] = trigger_vx * snap_heading[:, 1]

    z_hist, tilt_hist, fr_hist = [], [], []
    with torch.no_grad():
        for t in range(DIVE_STEPS):
            obs, _, _, _ = dw.step(dpol(obs))
            z, tu, _, fr = read_state(dbase, frontier=True)
            z_hist.append(z.cpu()); tilt_hist.append(tu.cpu()); fr_hist.append(fr.cpu())
    dbase.close()

    zH = torch.stack(z_hist)      # [T, N]
    tH = torch.stack(tilt_hist)
    fH = torch.stack(fr_hist)
    eps = []
    for i in range(n):
        fmax = float(fH[:, i].max())
        gate = next((t for t in range(fH.shape[0]) if fH[t, i] >= math.radians(260)), None)
        rise = None
        t0 = gate if gate is not None else 0
        ok = zH[:, i] >= RISE_Z
        for t in range(t0, zH.shape[0] - RISE_SUSTAIN + 1):
            if bool(ok[t : t + RISE_SUSTAIN].all()):
                rise = t
                break
        fz = float(zH[-1, i])
        ft = math.degrees(math.acos(max(-1.0, min(1.0, float(tH[-1, i])))))
        eps.append({
            "frontier_max_deg": round(math.degrees(fmax), 1),
            "roll_complete": bool(fmax >= math.radians(260)),
            "full_2pi": bool(fmax >= math.radians(355)),
            "over_rotation": bool(fmax >= math.radians(385)),
            "kipup": rise is not None,
            "rise_step": rise,
            "final_z": round(fz, 4),
            "final_tilt_deg": round(ft, 1),
            "upright_final": bool(tH[-1, i] >= math.cos(math.radians(20))),
        })
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx-list", default="0.7,1.0,1.2")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--envs", type=int, default=8)
    args = ap.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    all_results = {}
    for vx in [float(x) for x in args.vx_list.split(",")]:
        eps = []
        for b in range(args.batches):
            eps += run_batch(vx, seed=100 + b * 17, num_envs=args.envs, device=device)
            print(f"[vx={vx}] batch {b}: {len(eps)} rollouts so far", flush=True)
        n = len(eps)
        def pct(key):
            return round(100.0 * sum(e[key] for e in eps) / n, 1)
        summary = {
            "n": n,
            "roll_complete_pct": pct("roll_complete"),
            "full_2pi_pct": pct("full_2pi"),
            "over_rotation_pct": pct("over_rotation"),
            "kipup_pct": pct("kipup"),
            "upright_final_pct": pct("upright_final"),
            "median_frontier_deg": sorted(e["frontier_max_deg"] for e in eps)[n // 2],
            "median_final_z": sorted(e["final_z"] for e in eps)[n // 2],
            "median_final_tilt_deg": sorted(e["final_tilt_deg"] for e in eps)[n // 2],
        }
        all_results[str(vx)] = {"summary": summary, "episodes": eps}
        print(f"\n=== trigger vx = {vx} m/s (n={n}) ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\n[handoff_eval_vx] saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
