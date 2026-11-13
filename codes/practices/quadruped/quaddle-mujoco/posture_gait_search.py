"""Posture-stabilized CPG gait search (classical control, no RL).

Adds body pitch/roll PD feedback to the leg targets on top of the duty CPG,
then random-searches gait + feedback gains for stable, smooth, fast walking
at servo kp=2.0. Objective: no fall, high speed, low vertical bounce, low
attitude variance (upright).
"""
import os, json, math, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from pathlib import Path

HERE = Path("/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco")
C = json.loads((HERE / "gait.json").read_text())
S = np.array(C["actuator_signs"]); B = np.array(C["bias_rad"])
A = np.array(C["amplitude_rad"]); P0 = np.array(C["phase_rad"])
RAMP = C["ramp_seconds"]
# leg order: lf, rf, rb, lb ; front={0,1} back={2,3} left={0,3} right={1,2}
FRONT = np.array([1, 1, -1, -1]); LEFT = np.array([1, -1, -1, 1])


def wave(ph, duty):
    c = (ph / (2 * math.pi)) % 1.0
    return np.cos(np.where(c < duty, math.pi * c / duty, math.pi + math.pi * (c - duty) / (1 - duty)))


def trial(freq, ascale, duty, kpp, kpr, kdp, kp=2.0, dur=10, scene="models/scene.xml"):
    m = mujoco.MjModel.from_xml_path(str(HERE / scene)); m.opt.timestep = 0.001
    m.actuator_gainprm[:, 0] = kp; m.actuator_forcerange[:] = [-4, 4]
    d = mujoco.MjData(m); bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "body")
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    for _ in range(2000): mujoco.mj_step(m, d)
    p0 = d.qpos[:2].copy(); amp = 0.0; ph = 0.0
    zs = []; pitch = []; roll = []; fell = False
    for f in range(int(dur / 0.1)):
        for _ in range(5):
            amp = min(1, amp + 0.02 / RAMP); ph += 2 * math.pi * freq * 0.02
            r = d.body(bid).xmat.reshape(3, 3)
            pi_ = math.atan2(-r[2, 0], r[2, 2]); ro_ = math.atan2(r[2, 1], r[2, 2])
            pir = d.qvel[4]
            corr = kpp * pi_ * FRONT + kpr * ro_ * LEFT + kdp * pir * FRONT
            w = wave(ph + P0, np.full(4, duty))
            d.ctrl[:] = np.clip(S * (B + A * ascale * amp * w) + corr, -1.2, 1.2)
            for _ in range(20): mujoco.mj_step(m, d)
        z = d.body(bid).xpos[2]; zs.append(z)
        r = d.body(bid).xmat.reshape(3, 3)
        pitch.append(math.atan2(-r[2, 0], r[2, 2])); roll.append(math.atan2(r[2, 1], r[2, 2]))
        if z < 0.032: fell = True; break
    if fell: return None
    disp = np.linalg.norm(d.qpos[:2] - p0); sp = disp / dur * 1000
    bo = np.std(zs) * 1000
    att = np.std(pitch) * 180 / math.pi + np.std(roll) * 180 / math.pi
    return dict(speed=sp, bounce=bo, att=att, freq=freq, ascale=ascale, duty=duty,
                kpp=kpp, kpr=kpr, kdp=kdp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = []
    for t in range(args.trials):
        r = trial(rng.uniform(1.0, 2.6), rng.uniform(0.5, 1.1), rng.uniform(0.5, 0.75),
                  rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.05, 0.05))
        if r: res.append(r)
    # score: speed rewarded, bounce & attitude penalized
    for r in res:
        r["score"] = r["speed"] - 3 * r["bounce"] - 2 * r["att"]
    res.sort(key=lambda r: -r["score"])
    print(f"stable {len(res)}/{args.trials}; TOP by score:")
    for r in res[:15]:
        print(f"score={r['score']:6.1f} speed={r['speed']:6.1f} bounce={r['bounce']:5.2f} att={r['att']:5.2f} "
              f"freq={r['freq']:.2f} ascale={r['ascale']:.2f} duty={r['duty']:.2f} "
              f"kpp={r['kpp']:.3f} kpr={r['kpr']:.3f} kdp={r['kdp']:.3f}", flush=True)
    json.dump(res[:15], open("/root/posture_top.json", "w"), indent=1)