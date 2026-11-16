"""L4 conditional/sequential eval.
  (a) "stop when you see green": success if robot halts near green.
  (b) "go to red then blue": success if red reached then blue reached in order.
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--out", default=str(HERE / "outputs/rollout_l4.gif"))
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
    model = mujoco.MjModel.from_xml_path(str(HERE / "outputs/dataset_l4/scene.xml"))
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
        task_type = ep % 2
        layout = cv.random_layout(rng)
        pos_of = {c: layout[i] for i, c in enumerate(cv.COLORS)}
        lang = "stop when you see green" if task_type == 0 else "go to red then blue"
        targets = ["green"] if task_type == 0 else ["red", "blue"]
        yaw0 = float(rng.uniform(-math.pi, math.pi))
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        for i, c in enumerate(cv.COLORS):
            model.geom_pos[geom_ids[c]][:2] = list(layout[i])
        data.ctrl[:] = 0; mujoco.mj_forward(model, data)
        for _ in range(int(cv.SETTLE_S / cv.DT_PHYS)): mujoco.mj_step(model, data)
        amp_scale = phases_t = 0.0
        skill0 = None; last_switch_t = 0.0
        chunk = None; chunk_i = 0
        stage = 0; reached = [False] * len(targets)
        success = fell = False; t = 0.0
        ep_frames = []
        while t < args.max_s:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            tgt = pos_of[targets[stage]]
            dist = float(np.linalg.norm(tgt - pos[:2]))
            d_body = r.T @ np.array([tgt[0] - pos[0], tgt[1] - pos[1], 0.0])
            bearing = math.atan2(d_body[0], -d_body[1])
            if chunk is None or chunk_i >= 5:
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
                    "task": [lang],
                }
                batch = pre(batch)
                with torch.no_grad():
                    chunk = policy.select_action(batch)
                chunk_i = 0
            act = chunk[0].float().cpu().numpy(); chunk_i += 1
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
            if task_type == 0:
                if dist < 0.20 and skill == "stop": success = True; break
            else:
                if dist < cv.SUCCESS_DIST:
                    reached[stage] = True
                    if stage < len(targets) - 1: stage += 1
                    else: success = True; break
        results.append(success)
        print(f"ep{ep}: {lang} {'✓' if success else '✗'} reached={reached} fell={fell}", flush=True)
        all_frames.extend(ep_frames)
    imageio.mimsave(args.out, all_frames, fps=10, loop=0)
    n_ok = sum(results)
    print(f"L4_SUCCESS {n_ok}/{len(results)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()