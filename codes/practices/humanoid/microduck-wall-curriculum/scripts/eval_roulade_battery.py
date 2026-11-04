"""Headless eval battery for a roulade checkpoint (AGENTS.md: measure before theorizing).

Per-spawn-type batteries (standing / supine / seated / prone-head), deterministic
policy, one episode each (250 steps @ 50 Hz). Reports roll completion, rise
latency after the landing gate opens, rise speed, end-state quality — the
numbers the "slow kip-up" (鲤鱼打挺) diagnosis needs.

Usage:
  .venv/bin/python scripts/eval_roulade_battery.py \
      --checkpoint logs/rsl_rl/microduck_roulade/<run>/model_4999.pt \
      [--num-envs 64] [--modes train,play] [--batteries standing,supine,seated,prone]
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (populate registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-Roulade-Flat-MicroDuck"
STAND_Z = 0.115
GATE_LO = math.radians(260.0)
GATE_HI = math.radians(330.0)
OPEN_AT = math.radians(300.0)  # "gate effectively open" crossing
RISE_Z = 0.9 * STAND_Z
SUSTAIN = 25  # steps (0.5 s) z must stay above RISE_Z to count as risen
HORIZON = 250  # 5 s episode at 50 Hz

BATTERIES = {
    # full trick from a standing start: roll over the head, land, rise
    "standing": {"standing_prob": 1.0, "midroll_prob": 0.0},
    # classic kip-up start: on the back, some forward momentum
    "supine": {
        "standing_prob": 0.0,
        "midroll_prob": 1.0,
        "midroll_pitch_min": math.radians(170.0),
        "midroll_pitch_max": math.radians(230.0),
        "midroll_omega_range": (1.0, 3.0),
    },
    # late-roll spawn: seated leaning back, gate open from birth
    "seated": {
        "standing_prob": 0.0,
        "midroll_prob": 1.0,
        "midroll_pitch_min": math.radians(280.0),
        "midroll_pitch_max": math.radians(340.0),
        "midroll_omega_range": (0.0, 2.0),
    },
    # early-roll spawn: face/head down, must finish the roll first
    "prone": {
        "standing_prob": 0.0,
        "midroll_prob": 1.0,
        "midroll_pitch_min": math.radians(50.0),
        "midroll_pitch_max": math.radians(120.0),
        "midroll_omega_range": (1.0, 3.0),
    },
}


def read_tuple(env_base):
    """(z, tilt_up, vz, frontier, latch) tensors — env-0 diagnostics."""
    robot = env_base.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2] - env_base.scene.terrain.env_origins[:, 2]
    quat = robot.data.root_link_quat_w
    tilt_up = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    vz = torch.nan_to_num(robot.data.root_link_lin_vel_w[:, 2], nan=0.0)
    frontier = torch.nan_to_num(env_base._roulade_max.clone(), nan=0.0)
    latch = env_base._roulade_head_latch.clone()
    return z, tilt_up, vz, frontier, latch


def _pct(vals: list, q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def run_battery(env_base, policy, name: str, num_envs: int, latch_required: bool):
    """Force spawn params via the EVENT MANAGER (never env.cfg — silent no-op), run one episode."""
    tc = env_base.event_manager.get_term_cfg("set_roulade_state")
    defaults = {k: tc.params[k] for k in BATTERIES[name]}
    try:
        for k, v in BATTERIES[name].items():
            tc.params[k] = v
        obs, _ = env_base.reset()
    finally:
        for k, v in defaults.items():
            tc.params[k] = v

    def read_state():
        robot = env_base.scene["robot"]
        z = robot.data.root_link_pos_w[:, 2] - env_base.scene.terrain.env_origins[:, 2]
        quat = robot.data.root_link_quat_w
        tilt_up = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))  # cos(tilt)
        vz = torch.nan_to_num(robot.data.root_link_lin_vel_w[:, 2], nan=0.0)
        frontier = torch.nan_to_num(env_base._roulade_max.clone(), nan=0.0)
        latch = env_base._roulade_head_latch.clone()
        found = env_base.scene.sensors["feet_ground_contact"].data.found
        air = ~(found.view(found.shape[0], -1) > 0).any(dim=-1)
        return z, tilt_up, vz, frontier, latch, air

    cols = {"z": [], "tilt_up": [], "vz": [], "frontier": [], "latch": [], "air": [], "done": []}
    # state at index t is read BEFORE step t — so the state at the first done
    # index is the true episode-end state, not the auto-reset spawn pose.
    with torch.no_grad():
        for t in range(HORIZON):
            z, tilt_up, vz, frontier, latch, air = read_state()
            cols["z"].append(z.cpu()); cols["tilt_up"].append(tilt_up.cpu())
            cols["vz"].append(vz.cpu()); cols["frontier"].append(frontier.cpu())
            cols["latch"].append(latch.cpu()); cols["air"].append(air.cpu())
            actions = policy(obs)
            obs, _, terminated, time_out, _ = env_base.step(actions)
            cols["done"].append((terminated | time_out).cpu())

    # stack → [T, N]; freeze each env at its first done (auto-reset pollutes after)
    T = len(cols["z"])
    z = torch.stack(cols["z"]); tilt = torch.stack(cols["tilt_up"])
    vz = torch.stack(cols["vz"]); frontier = torch.stack(cols["frontier"])
    latch = torch.stack(cols["latch"]).to(torch.bool)
    air = torch.stack(cols["air"]).to(torch.bool)
    done = torch.stack(cols["done"]).to(torch.bool)
    N = z.shape[1]

    episodes = []
    for i in range(N):
        d = int(done[:, i].float().argmax()) + 1 if done[:, i].any() else T  # first done → episode length
        zE, tE, vzE, fE, lE, aE = (x[:d, i] for x in (z, tilt, vz, frontier, latch, air))
        fmax = float(fE.max()) if len(fE) else 0.0
        latched = bool(lE.any())
        gate_open = next((t for t in range(len(fE)) if fE[t] >= OPEN_AT), None)
        gate_val = float(latched) if latch_required else 1.0
        roll_done = fmax >= GATE_LO and (latched or not latch_required)
        # Rise = first SUSTAINED z ≥ RISE_Z window at/after gate_open — searching
        # from t=0 misfires on standing spawns that start the fold slowly
        # (z stays above RISE_Z pre-roll → rise=0 < gate_open → vz window empty).
        rise = None
        t0 = gate_open if gate_open is not None else 0
        if len(zE) - t0 >= SUSTAIN:
            ok = zE >= RISE_Z
            for t in range(t0, len(zE) - SUSTAIN + 1):
                if bool(ok[t : t + SUSTAIN].all()):
                    rise = t
                    break
        air_post = aE[t0:]  # flight steps in the post-roll window
        air_steps = int(air_post.sum()) if len(air_post) else 0
        air_max_vz = float(vzE[t0:][air_post].max()) if air_steps else 0.0
        lat = None if (rise is None or gate_open is None) else rise - gate_open
        max_vz = float(vzE[gate_open : rise + 1].max()) if (gate_open is not None and rise is not None and rise > gate_open) else 0.0
        episodes.append({
            "env": i, "steps_alive": d,
            "frontier_max_deg": round(math.degrees(fmax), 1),
            "head_latch": latched,
            "roll_complete": bool(roll_done),
            "gate_open_step": gate_open,
            "rise_step": rise,
            "rise_latency_steps": lat,
            "rise_max_vz": round(max_vz, 3),
            "airborne_steps_post_gate": air_steps,
            "airborne_max_vz": round(air_max_vz, 3),
            "final_z": round(float(zE[-1]), 4) if len(zE) else None,
            "final_tilt_deg": round(math.degrees(math.acos(max(-1, min(1, float(tE[-1]))))), 1) if len(tE) else None,
            "final_upright_ok": bool(len(tE) and tE[-1] >= math.cos(math.radians(20.0))),
        })

    rose = [e for e in episodes if e["rise_latency_steps"] is not None]
    lats = [e["rise_latency_steps"] for e in rose]
    summary = {
        "n": len(episodes),
        "roll_complete_pct": 100.0 * sum(e["roll_complete"] for e in episodes) / len(episodes),
        "full_2pi_pct": 100.0 * sum(e["frontier_max_deg"] >= 355 for e in episodes) / len(episodes),
        "head_latch_pct": 100.0 * sum(e["head_latch"] for e in episodes) / len(episodes),
        "rise_pct": 100.0 * len(rose) / len(episodes),
        "latency_p50_s": round(_pct(lats, 0.5) * 0.02, 2) if lats else None,
        "latency_p90_s": round(_pct(lats, 0.9) * 0.02, 2) if lats else None,
        "kipup_le_1s_pct": 100.0 * sum(1 for L in lats if L <= 50) / len(episodes),
        "kipup_le_2s_pct": 100.0 * sum(1 for L in lats if L <= 100) / len(episodes),
        "rise_max_vz_p50": round(_pct([e["rise_max_vz"] for e in rose], 0.5), 3) if rose else None,
        "airborne_pct": 100.0 * sum(1 for e in episodes if e["airborne_steps_post_gate"] > 0) / len(episodes),
        "airborne_steps_p50": round(_pct([e["airborne_steps_post_gate"] for e in episodes], 0.5), 1),
        "airborne_max_vz_p50": round(_pct([e["airborne_max_vz"] for e in episodes if e["airborne_steps_post_gate"] > 0], 0.5), 3) if any(e["airborne_steps_post_gate"] > 0 for e in episodes) else 0.0,
        "final_upright_pct": 100.0 * sum(e["final_upright_ok"] for e in episodes) / len(episodes),
        "median_final_z": _pct([e["final_z"] for e in episodes if e["final_z"] is not None], 0.5),
        "median_final_tilt_deg": _pct([e["final_tilt_deg"] for e in episodes if e["final_tilt_deg"] is not None], 0.5),
        "alive_pct": 100.0 * sum(e["steps_alive"] >= HORIZON - 1 for e in episodes) / len(episodes),
    }
    return name, {"summary": summary, "episodes": episodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", type=str, default=TASK)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--batteries", type=str, default="standing,supine,seated,prone")
    ap.add_argument("--modes", type=str, default="train")
    ap.add_argument("--steps", type=int, default=HORIZON)
    ap.add_argument("--trace", action="store_true", help="print env-0 z/tilt/frontier every 25 steps")
    ap.add_argument("--emulate-video", action="store_true", help="render_mode=rgb_array + VideoRecorder, like video_rollout.py")
    ap.add_argument("--infer-mode", type=str, default="no_grad", choices=["no_grad", "inference_mode"])
    ap.add_argument("--camera", type=str, default=None, help="distance,elevation,azimuth — viewer mutations like video_rollout flags")
    ap.add_argument("--dump", type=int, default=0, help="dump full obs/action vectors for first N steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent_cfg = load_rl_cfg(args.task)

    results = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        env_cfg = load_env_cfg(args.task, play=(mode == "play"))
        env_cfg.scene.num_envs = args.num_envs
        if args.camera:
            dist, ele, azi = (float(x) for x in args.camera.split(","))
            env_cfg.viewer.distance = dist
            env_cfg.viewer.elevation = ele
            env_cfg.viewer.azimuth = azi
        env_base = ManagerBasedRlEnv(
            cfg=env_cfg, device=device, render_mode="rgb_array" if args.emulate_video else None
        )
        if args.emulate_video:
            from mjlab.utils.wrappers import VideoRecorder
            env = VideoRecorder(
                env_base,
                video_folder=Path(args.checkpoint).parent / "videos" / "play",
                step_trigger=lambda step: step == 0,
                video_length=args.steps,
                disable_logger=True,
            )
        else:
            env = env_base
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(args.checkpoint, map_location=device)
        policy = runner.get_inference_policy(device=device)

        if args.batteries.strip().lower() == "none":
            if args.seed is not None:
                env_base.seed(args.seed)
            obs, _ = env.reset()
            ctx = torch.inference_mode() if args.infer_mode == "inference_mode" else torch.no_grad()
            n_done = 0
            with ctx:
                for t in range(args.steps):
                    if args.trace and t % 25 == 0:
                        actor_obs = obs["actor"] if hasattr(obs, "keys") else obs
                        z, tilt, vz, fr, la = read_tuple(env_base)
                        print(
                            f"  t={t:3d} z={z[0].item():.4f} tilt={math.degrees(math.acos(max(-1.0, min(1.0, tilt[0].item())))):5.1f}deg"
                            f" frontier={math.degrees(fr[0].item()):6.1f}deg latch={bool(la[0].item())}"
                            f" | obs[{actor_obs[0,0].item():+.2f},{actor_obs[0,1].item():+.2f},{actor_obs[0,30].item():+.2f}…] μ={actor_obs.mean().item():+.3f} σ={actor_obs.std().item():.3f}"
                        )
                    a = policy(obs)
                    if args.trace and t % 25 == 0:
                        print(f"        action μ={a.mean().item():+.4f} σ={a.std().item():.4f} ‖a‖={a.norm().item():.3f}")
                    if args.dump and t < args.dump:
                        ao = obs["actor"] if hasattr(obs, "keys") else obs
                        print(f"[DBG] t={t} obs: {ao[0].cpu().numpy().round(4).tolist()}")
                        print(f"[DBG] t={t} act: {a[0].cpu().numpy().round(4).tolist()}")
                    step_out = env.step(a)
                    obs = step_out[0]
            print(f"[none-battery] {mode}: {args.steps} steps done")
            env.close()
            continue

        for b in [b.strip() for b in args.batteries.split(",") if b.strip()]:
            # standing spawns must earn the head latch; midroll spawns have it at birth
            latch_required = b == "standing"
            name, out = run_battery(env_base, policy, b, args.num_envs, latch_required)
            results[f"{mode}/{name}"] = out
            s = out["summary"]
            print(f"\n=== {mode}/{name} ===")
            for k, v in s.items():
                print(f"  {k}: {v}")
        env.close()

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / f"battery_{Path(args.checkpoint).stem}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[eval_roulade_battery] saved -> {out_path}")


if __name__ == "__main__":
    main()
