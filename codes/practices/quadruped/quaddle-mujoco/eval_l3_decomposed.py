"""L3 decomposed eval: L2 nav skill + obstacle reflex layer (no L3 ckpt).

Breaks the L3 architecture ceiling (end-to-end 2/20) by subsumption:
  - Mover  = the 85% L2 skill (18k ckpt + temporal ensembling, native phrase
    "walk to the {color} block"), unchanged.
  - Reflex = when the nearest NON-target block is within --reflex-r of the body
    and roughly ahead (|bearing| < 100 deg), override with a steer-away gear
    for --reflex-dwell seconds, then resume the skill with a fresh ensemble
    buffer. Obstacle/target identity comes from the same scene-state interface
    the policy's target state (state[6:8]) is built on.

Success/collision criteria and RNG/scene protocol are byte-identical to
eval_l3.py (reach target <12cm, no non-target block within 9cm, no fall).
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import collect_v5 as cv
from eval_rollout_v5 import quantize

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--ens-k", type=float, default=0.01)
    ap.add_argument("--reflex-r", type=float, default=0.15, help="trigger radius (collision judged at 0.09)")
    ap.add_argument("--reflex-dwell", type=float, default=0.8, help="steer-away hold time (s)")
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--out", default=str(HERE / "outputs/rollout_l3_decomposed.gif"))
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch, mujoco, imageio.v2 as imageio
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt); policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device=device)
    model = mujoco.MjModel.from_xml_path(str(HERE / "outputs/dataset_l3/scene.xml"))
    model.opt.timestep = cv.DT_PHYS
    data = mujoco.MjData(model)
    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in cv.COLORS}
    renderer = mujoco.Renderer(model, height=cv.IMG, width=cv.IMG)
    gif_cam = mujoco.MjvCamera(); gif_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    gif_cam.lookat[:] = [0, 0, 0]; gif_cam.distance = 1.3; gif_cam.azimuth = 135; gif_cam.elevation = -35
    rng = np.random.default_rng(args.seed)
    all_frames, results = [], []
    for ep in range(args.episodes):
        color = cv.COLORS[ep % 3]
        layout = cv.random_layout(rng)
        tgt_xy = layout[cv.COLORS.index(color)]
        obs_xy = [layout[i] for i in range(3) if i != cv.COLORS.index(color)]
        yaw0 = float(rng.uniform(-math.pi, math.pi))
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        for i, c in enumerate(cv.COLORS):
            model.geom_pos[geom_ids[c]][:2] = list(layout[i])
        data.ctrl[:] = 0; mujoco.mj_forward(model, data)
        for _ in range(int(cv.SETTLE_S / cv.DT_PHYS)): mujoco.mj_step(model, data)
        amp_scale = phases_t = 0.0
        skill0 = None; last_switch_t = 0.0
        chunks = []; frame = 0; force_replan = True
        avoid_skill = None; reflex_until = -1.0
        success = fell = collided = False; t = 0.0
        ep_frames = []
        while t < args.max_s:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            dist = float(np.linalg.norm(tgt_xy - pos[:2]))
            for ox, oy in obs_xy:
                if np.linalg.norm(np.array([ox, oy]) - pos[:2]) < 0.09: collided = True
            if collided: break
            d_body = r.T @ np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
            bearing = math.atan2(d_body[0], -d_body[1])
            # ---- 反射层：最近非目标块近距在前 -> 转向避让（子胜层，优先于技能输出）----
            if t >= reflex_until and avoid_skill is not None:
                avoid_skill = None; force_replan = True   # 避让结束，恢复技能并清集成缓冲
            if avoid_skill is None:
                best = None
                for ox, oy in obs_xy:
                    od = float(np.linalg.norm(np.array([ox, oy]) - pos[:2]))
                    if od < args.reflex_r and (best is None or od < best[0]):
                        ob = r.T @ np.array([ox - pos[0], oy - pos[1], 0.0])
                        b_obs = math.atan2(ob[0], -ob[1])
                        best = (od, b_obs)
                if best is not None and abs(best[1]) < math.radians(100):
                    avoid_skill = "steer_R" if best[1] > 0 else "steer_L"   # 背向障碍转
                    reflex_until = t + args.reflex_dwell
                    chunks = []; skill0 = None; last_switch_t = t
            if (avoid_skill is None) and (force_replan or frame % args.replan_steps == 0):
                imgs = {}
                for cam in ("overhead", "front", "oblique"):
                    renderer.update_scene(data, camera=cam); imgs[cam] = renderer.render().copy()
                state = np.zeros(8, dtype=np.float32)
                state[:4] = [data.qpos[servo_qposadr[i]] for i in range(4)]
                v_body = r.T @ data.qvel[:3]
                state[4] = float(-v_body[1]); state[5] = float((r.T @ data.qvel[3:6])[2])
                state[6] = bearing; state[7] = dist
                batch = {
                    "observation.images.camera1": torch.from_numpy(imgs["overhead"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.camera2": torch.from_numpy(imgs["front"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.camera3": torch.from_numpy(imgs["oblique"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.state": torch.from_numpy(state).unsqueeze(0),
                    "task": [f"walk to the {color} block"],
                }
                batch = pre(batch)
                with torch.no_grad():
                    ch = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                chunks.append((frame, ch))
                chunks = chunks[-12:]
                force_replan = False
            if avoid_skill is not None:
                skill = avoid_skill
            else:
                num = np.zeros(2); den = 0.0
                for s, a in chunks:
                    idx = frame - s
                    if 0 <= idx < len(a):
                        w = math.exp(-args.ens_k * idx)
                        num += w * a[idx]; den += w
                act = num / den if den > 0 else np.zeros(2)
                skill, _ = quantize(act)
                if skill != skill0 and skill0 is not None and t - last_switch_t < cv.DWELL_S and skill != "stop":
                    skill = skill0
            steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
            amp_scale = min(1.0, amp_scale + cv.DT_CTRL / cv.RAMP_S)
            phases_t += 2 * math.pi * cv.FREQ * cv.DT_CTRL
            duty = np.full(4, cv.DUTY)
            if steer_now > 0:
                duty[list(cv.SIDE["L"])] *= (1 - steer_now * cv.STEER_LIMIT); duty[list(cv.SIDE["R"])] *= (1 + steer_now * cv.STEER_LIMIT)
            elif steer_now < 0:
                duty[list(cv.SIDE["L"])] *= (1 + abs(steer_now) * cv.STEER_LIMIT); duty[list(cv.SIDE["R"])] *= (1 - abs(steer_now) * cv.STEER_LIMIT)
            w = cv.wave_at(phases_t + cv.PHASE0, duty)
            data.ctrl[:] = np.clip(cv.SIGNS * (cv.BIAS + cv.AMPS * amp_scale * w), -cv.CTRL_MAX, cv.CTRL_MAX)
            if skill != skill0: skill0, last_switch_t = skill, t
            for _ in range(5):
                for _ in range(cv.SUB): mujoco.mj_step(model, data)
                t += cv.DT_CTRL
                if data.body(body_id).xpos[2] < 0.030: fell = True; break
            if fell: break
            renderer.update_scene(data, camera=gif_cam); ep_frames.append(renderer.render().copy())
            frame += 1
            if dist < cv.SUCCESS_DIST: success = True; break
        ok = success and not collided
        results.append(ok)
        print(f"ep{ep}: {color} {'✓' if ok else '✗'} success={success} collided={collided} fell={fell} dist={dist*100:.0f}cm", flush=True)
        all_frames.extend(ep_frames)
    imageio.mimsave(args.out, all_frames, fps=10, loop=0)
    n_ok = sum(results)
    print(f"L3_SUCCESS {n_ok}/{len(results)} (target>=50%, end-to-end baseline 2/20)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
