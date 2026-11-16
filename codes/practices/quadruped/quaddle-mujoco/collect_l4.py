"""L4 conditional/sequential task collection.

Two task types:
  (a) "stop when you see green": move forward; stop once green block is near.
  (b) "go to red then blue": navigate to red (<12cm), then to blue, then stop.
Expert executes the conditional/sequential logic; records (obs, language, action).
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import collect_v5 as cv

HERE = Path(__file__).resolve().parent
ACTIONS = {"forward": (1.0, 0.0), "steer_L": (0.0, 1.0), "steer_R": (0.0, -1.0), "stop": (0.0, 0.0)}


def nav_skill(bearing, dist, deadzone):
    if dist < cv.SUCCESS_DIST: return "stop"
    if abs(bearing) > deadzone: return "steer_L" if bearing > 0 else "steer_R"
    return "forward"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--repo-id", default="local/quaddle_l4")
    ap.add_argument("--root", default=str(HERE / "outputs/dataset_l4"))
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--rear-oversample", type=float, default=0.5)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from lerobot.datasets import LeRobotDataset

    out_dir = Path(args.root); out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = out_dir / "scene.xml"; cv.build_scene(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml)); model.opt.timestep = cv.DT_PHYS
    data = mujoco.MjData(model)
    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in cv.COLORS}
    renderer = mujoco.Renderer(model, height=cv.IMG, width=cv.IMG)

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
    while n_saved < args.episodes and ep < args.episodes * 6:
        ep += 1
        task_type = 0 if rng.random() < 0.6 else 1  # 60% stop-at-green, 40% sequential
        layout = cv.random_layout(rng)
        pos_of = {c: layout[i] for i, c in enumerate(cv.COLORS)}
        if task_type == 0:
            lang = "stop when you see green"
            targets = ["green"]
        else:
            lang = "go to red then blue"
            targets = ["red", "blue"]
        th_t = math.atan2(pos_of[targets[0]][1], pos_of[targets[0]][0])
        if rng.random() < args.rear_oversample:
            yaw0 = th_t - math.pi / 2 + rng.uniform(-math.radians(30), math.radians(30))
        else:
            yaw0 = float(rng.uniform(-math.pi, math.pi))
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        for i, c in enumerate(cv.COLORS):
            model.geom_pos[geom_ids[c]][:2] = list(layout[i])
        data.ctrl[:] = 0; mujoco.mj_forward(model, data)
        for _ in range(int(cv.SETTLE_S / cv.DT_PHYS)): mujoco.mj_step(model, data)

        deadzone = float(rng.uniform(cv.BEARING_DEADZONE_MIN, cv.BEARING_DEADZONE_MAX))
        amp_scale = phases_t = 0.0
        skill0 = None; last_switch_t = 0.0
        stage = 0
        frames = []; success = fell = False; t = 0.0
        while t < cv.TIMEOUT_S:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            tgt = pos_of[targets[stage]]
            d_world = np.array([tgt[0] - pos[0], tgt[1] - pos[1], 0.0])
            dist = float(np.linalg.norm(d_world[:2]))
            d_body = r.T @ d_world
            bearing = math.atan2(d_body[0], -d_body[1])
            if task_type == 0:
                # stop when green near
                skill = "stop" if dist < 0.20 else nav_skill(bearing, dist, deadzone)
                if dist < 0.20: success = True
            else:
                skill = nav_skill(bearing, dist, deadzone)
                if dist < cv.SUCCESS_DIST:
                    if stage < len(targets) - 1:
                        stage += 1
                    else:
                        success = True
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
                "action": np.array(ACTIONS[skill], dtype=np.float32),
                "task": lang,
            })
            if success: break
        if success and frames:
            for fr in frames: dataset.add_frame(fr)
            dataset.save_episode(); n_saved += 1
            if n_saved % 40 == 0: print(f"L4 saved {n_saved}/{args.episodes}", flush=True)
        else:
            n_fail += 1
    dataset.finalize()
    print(f"L4_DONE saved={n_saved} fail={n_fail} -> {ds_root}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()