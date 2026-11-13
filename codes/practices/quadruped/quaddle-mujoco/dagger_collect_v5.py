"""DAgger closed-loop recovery collection (VLA, no RL).

Roll out the CURRENT policy (with replan) so it visits its own closed-loop states
(including OOD self-spin states), but label each frame with the EXPERT's action
computed from the privileged bearing/dist. Retraining on these policy-visited
states with expert recovery labels directly targets compounding error.

Reuses collect_v5 scene/expert/executor and eval_rollout_v5 quantize.
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
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--repo-id", default="local/quaddle_nav_dagger")
    ap.add_argument("--root", default=str(HERE / "outputs/dataset_dagger"))
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--replan-steps", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import torch, mujoco
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    out_dir = Path(args.root); out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = out_dir / "scene.xml"; cv.build_scene(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml)); model.opt.timestep = cv.DT_PHYS
    data = mujoco.MjData(model)
    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in cv.COLORS}
    renderer = mujoco.Renderer(model, height=cv.IMG, width=cv.IMG)

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt); policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device=device)

    features = {f"observation.images.camera{i}": {"dtype": "video", "shape": (cv.IMG, cv.IMG, 3),
                "names": ["height", "width", "channel"]} for i in (1, 2, 3)}
    features["observation.state"] = {"dtype": "float32", "shape": (8,), "names": ["state"]}
    features["action"] = {"dtype": "float32", "shape": (2,), "names": ["action"]}
    ds_root = out_dir / "lerobot"
    if ds_root.exists():
        import shutil; shutil.rmtree(ds_root)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=cv.DATA_HZ, features=features,
                                    root=str(ds_root), robot_type="quaddle", use_videos=True)

    rng = np.random.default_rng(args.seed)
    n_saved = n_fail = ep = 0
    ACTIONS = {"forward": (1.0, 0.0), "steer_L": (0.0, 1.0), "steer_R": (0.0, -1.0), "stop": (0.0, 0.0)}

    while n_saved < args.episodes and ep < args.episodes * 6:
        ep += 1
        color = cv.COLORS[ep % 3]
        deadzone = float(rng.uniform(cv.BEARING_DEADZONE_MIN, cv.BEARING_DEADZONE_MAX))
        layout = cv.random_layout(rng)
        tgt_xy = layout[cv.COLORS.index(color)]
        th_t = math.atan2(tgt_xy[1], tgt_xy[0])
        yaw0 = th_t - math.pi / 2 + rng.uniform(-math.radians(30), math.radians(30)) if rng.random() < cv.REAR_OVERSAMPLE else float(rng.uniform(-math.pi, math.pi))
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        for i, c in enumerate(cv.COLORS):
            model.geom_pos[geom_ids[c]][:2] = list(layout[i])
        data.ctrl[:] = 0; mujoco.mj_forward(model, data)
        for _ in range(int(cv.SETTLE_S / cv.DT_PHYS)): mujoco.mj_step(model, data)

        amp_scale = phases_t = 0.0
        skill0 = None; last_switch_t = 0.0
        chunk = None; chunk_i = 0
        frames = []; success = fell = False; t = 0.0
        while t < cv.TIMEOUT_S:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            dist = float(np.linalg.norm(tgt_xy - pos[:2]))
            d_body = r.T @ np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
            bearing = math.atan2(d_body[0], -d_body[1])
            # EXPERT label from current state (recovery action)
            if dist < cv.SUCCESS_DIST: exp_skill = "stop"
            elif abs(bearing) > deadzone: exp_skill = "steer_L" if bearing > 0 else "steer_R"
            else: exp_skill = "forward"
            # POLICY action for rollout (replan)
            if chunk is None or chunk_i >= args.replan_steps:
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
            # record frame with EXPERT label (DAgger)
            r = data.body(body_id).xmat.reshape(3, 3)
            v_body = r.T @ data.qvel[:3]
            state = np.concatenate([
                np.array([data.qpos[servo_qposadr[i]] for i in range(4)]),
                [float(-v_body[1]), float((r.T @ data.qvel[3:6])[2])],
                [bearing, dist]]).astype(np.float32)
            frames.append({
                "observation.images.camera1": cv.renderer_update(model, data, renderer, "overhead"),
                "observation.images.camera2": cv.renderer_update(model, data, renderer, "front"),
                "observation.images.camera3": cv.renderer_update(model, data, renderer, "oblique"),
                "observation.state": state,
                "action": np.array(ACTIONS[exp_skill], dtype=np.float32),
                "task": f"walk to the {color} block",
            })
            if dist < cv.SUCCESS_DIST: success = True; break
        # DAgger: save policy-visited states (incl. OOD/failure) with expert labels,
        # not only successful episodes — the OOD states are exactly the training signal.
        if frames:
            for fr in frames: dataset.add_frame(fr)
            dataset.save_episode(); n_saved += 1
            print(f"ep{ep:03d} {'✓' if success else '○'} {color} {len(frames)}帧 (saved {n_saved}/{args.episodes})", flush=True)
        else:
            n_fail += 1
            print(f"ep{ep:03d} ✗ {color} 无帧丢弃 (fail {n_fail})", flush=True)
    dataset.finalize()
    print(f"DAGGER_DONE saved={n_saved} fail={n_fail} -> {ds_root}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()