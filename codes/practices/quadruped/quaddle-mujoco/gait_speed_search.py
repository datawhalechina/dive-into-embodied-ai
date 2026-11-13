"""Gait speed+smoothness search (low-level CPG, no RL).

Maximize forward speed while minimizing vertical bounce and control jerk (silky),
subject to no fall. Varies freq / amplitude / servo kp / damping kv, with ctrl
slew-rate limiting for smooth transitions. Reports best smooth+fast configs.
"""
import os, json, math, argparse, itertools
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from pathlib import Path

HERE = Path("/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco")
C = json.loads((HERE / "gait.json").read_text())
S = np.array(C["actuator_signs"]); B = np.array(C["bias_rad"])
A = np.array(C["amplitude_rad"]); P0 = np.array(C["phase_rad"])
DUTY0 = C["sweep_fraction"]; RAMP = C["ramp_seconds"]
SIDE = {"L": (0, 3), "R": (1, 2)}


def wave(ph, duty):
    c = (ph / (2 * math.pi)) % 1.0
    return np.cos(np.where(c < duty, math.pi * c / duty, math.pi + math.pi * (c - duty) / (1 - duty)))


def trial(freq, ascale, kp, kv, slew, dur=6.0):
    m = mujoco.MjModel.from_xml_path(str(HERE / "models/scene.xml")); m.opt.timestep = 0.001
    m.actuator_gainprm[:, 0] = kp; m.actuator_biasprm[:, 1] = -kp; m.actuator_biasprm[:, 2] = -kv
    m.actuator_forcerange[:] = [-max(0.245, kp * 0.5), max(0.245, kp * 0.5)]
    d = mujoco.MjData(m); bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "body")
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    for _ in range(2000): mujoco.mj_step(m, d)
    p0 = d.qpos[:2].copy(); amp = 0.0; ph = 0.0; tp = np.zeros(4)
    zs = []; cj = []; fell = False
    for f in range(int(dur / 0.1)):
        for _ in range(5):
            amp = min(1, amp + 0.02 / RAMP); ph += 2 * math.pi * freq * 0.02
            w = wave(ph + P0, np.full(4, DUTY0))
            tgt = np.clip(S * (B + A * ascale * amp * w), -1.2, 1.2)
            if slew: tgt = tp + np.clip(tgt - tp, -slew * 0.02, slew * 0.02)
            tp = tgt; d.ctrl[:] = tgt
            for _ in range(20): mujoco.mj_step(m, d)
        z = d.body(bid).xpos[2]; zs.append(z); cj.append(d.ctrl.copy())
        if z < 0.032: fell = True; break
    if fell: return None
    disp = np.linalg.norm(d.qpos[:2] - p0); sp = disp / dur * 1000
    bo = np.std(zs) * 1000
    cj = np.array(cj); jerk = np.mean(np.std(np.diff(cj, n=2, axis=0), axis=0))
    return dict(speed=float(sp), bounce=float(bo), jerk=float(jerk), freq=float(freq),
                ascale=float(ascale), kp=float(kp), kv=float(kv), slew=int(slew))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    res = []
    for t in range(args.trials):
        r = trial(rng.uniform(1.5, 3.0), rng.uniform(0.8, 1.3), rng.uniform(0.15, 1.0),
                  rng.uniform(0.006, 0.06), rng.choice([0, 8, 16]))
        if r: res.append(r)
    for r in res:
        r["score"] = r["speed"] - 2.0 * r["bounce"] - 300 * r["jerk"]
    res.sort(key=lambda r: -r["score"])
    print(f"stable {len(res)}/{args.trials}; TOP smooth+fast:")
    for r in res[:12]:
        print(f"score={r['score']:6.1f} speed={r['speed']:6.1f} bounce={r['bounce']:5.2f} jerk={r['jerk']:.4f} "
              f"freq={r['freq']:.2f} ascale={r['ascale']:.2f} kp={r['kp']:.2f} kv={r['kv']:.3f} slew={r['slew']}", flush=True)
    fast = sorted([r for r in res if r["bounce"] < 1.5], key=lambda r: -r["speed"])
    print("TOP fastest (bounce<1.5mm):", flush=True)
    for r in fast[:8]:
        print(f"speed={r['speed']:6.1f} bounce={r['bounce']:5.2f} jerk={r['jerk']:.4f} "
              f"freq={r['freq']:.2f} ascale={r['ascale']:.2f} kp={r['kp']:.2f} kv={r['kv']:.3f} slew={r['slew']}", flush=True)
    json.dump({"smooth": res[:12], "fastest": fast[:8]}, open("/root/gait_speed_top.json", "w"), indent=1)