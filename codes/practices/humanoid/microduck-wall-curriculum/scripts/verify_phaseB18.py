"""B18 verification: cfg build asserts + midroll spawn distribution +
roulade_overhead_drive sanity rollout.

1. Build Mjlab-RouladeDiveWall-Flat-MicroDuck cfg with DIVEWALL_PHASE_B18 set
   (same env vars as train script) and assert the B18 additions are present.
2. Reset a small env fleet; check that ~frac of envs were midroll-spawned with
   accumulator in [165°, 270°] and are past the wall.
3. Step with zero actions a short while and print roulade_overhead_drive
   values (raw mdp call) — must be finite, in [0,1], and > 0 for some envs.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["DIVEWALL_STAGE"] = "4"
os.environ["DIVEWALL_WALL_SINK"] = "0.100"
os.environ["DIVEWALL_WALL_DIST"] = "0.40,0.65"
os.environ["DIVEWALL_POSTDIVE_PROB"] = "0.0"
os.environ["DIVEWALL_PHASE_B1"] = "0"
os.environ["DIVEWALL_PHASE_B3"] = "1"
os.environ["DIVEWALL_PHASE_B17"] = "0"
os.environ["DIVEWALL_PHASE_B18"] = "1"
os.environ["DIVEWALL_B18_MIDROLL_FRAC"] = "0.4"

import math
import torch
import mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"
N_ENVS = 64


def main():
    # ── 1. cfg assertions ──────────────────────────────────────────────
    cfg = load_env_cfg(TASK, play=True)
    r = cfg.rewards
    assert r["roulade_overhead_drive"].func.__name__ == "roulade_overhead_drive"
    assert r["roulade_overhead_drive"].weight == 1.5
    p = r["roulade_overhead_drive"].params
    assert abs(math.degrees(p["gate_lo"]) - 180.0) < 1e-6 and abs(math.degrees(p["gate_hi"]) - 240.0) < 1e-6
    assert p["rate_norm"] == 3.0 and p["wall_name"] == "wall"
    # B14 base intact
    assert r["roulade_progress"].weight == 12.0
    assert r["roulade_preroll_kick"].weight == 2.0
    assert r["roulade_stand_tax"].weight == 3.0
    # midroll spawn event, ordered AFTER reset_wall
    ev = cfg.events
    assert "midroll_spawn" in ev
    assert list(ev.keys()).index("reset_wall") < list(ev.keys()).index("midroll_spawn")
    mp = ev["midroll_spawn"].params
    assert mp["prob"] == 0.4
    assert mp["pitch_min_deg"] == 165.0 and mp["pitch_max_deg"] == 270.0
    assert mp["omega_range"] == (2.0, 5.0)
    print("[assert] B18 cfg OK: overhead_drive w=1.5 (180-240°), midroll_spawn 0.4 (165-270°, ω 2-5)")

    # ── 2. env build + reset: midroll spawn distribution ───────────────
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.num_envs = N_ENVS
    cfg.scene.num_envs = N_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array")
    env.reset()
    mdp = env.unwrapped if hasattr(env, "unwrapped") else env
    from mjlab_microduck.tasks import mdp as mdp_mod

    accum = env._roulade_accum
    lo, hi = math.radians(165.0), math.radians(270.0)
    mid = ((accum > lo - 0.05) & (accum <= hi + 0.01)).sum().item()
    frac = mid / N_ENVS
    print(f"[spawn] midroll envs: {mid}/{N_ENVS} ({frac:.2f}), accum range "
          f"[{math.degrees(accum.min()):.0f}°, {math.degrees(accum.max()):.0f}°]")
    assert 0.2 < frac < 0.6, "midroll fraction off"

    # ── 3. overhead_drive sanity: call the term directly ───────────────
    import sys
    sys.path.insert(0, "src")
    term = mdp_mod.roulade_overhead_drive
    vals = []
    adim = env.action_space.shape[-1]
    zero = torch.zeros(N_ENVS, adim, device=dev)
    for i in range(40):
        env.step(zero)
        v = term(env, gate_lo=math.radians(180.0), gate_hi=math.radians(240.0),
                 rate_norm=3.0, wall_name="wall")
        vals.append(v.detach())
    v = torch.stack(vals)  # [T, B]
    assert torch.isfinite(v).all(), "NaN in overhead_drive"
    assert (v >= 0).all() and (v <= 1.0 + 1e-6).all()
    nz = (v > 0.01).float().mean().item()
    print(f"[rollout] overhead_drive: mean={v.mean():.4f} max={v.max():.3f} "
          f"frac>0.01={nz:.3f} — finite, bounded, live")
    assert v.max() > 0.05, "overhead_drive never fires — check gates"
    print("[B18 verify] ALL PASS")


if __name__ == "__main__":
    main()
