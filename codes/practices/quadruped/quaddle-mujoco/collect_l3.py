"""L3 obstacle-avoidance navigation collection.

Task: walk to the {color} block while avoiding obstacle blocks placed in the path.
Expert = bearing-to-target control + obstacle repulsion (steer away when an obstacle
is close ahead). Records (obs, language, action). Eval: reach target <12cm without
colliding with obstacles (>=50%).
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import collect_v5 as cv

HERE = Path(__file__).resolve().parent
ACTIONS = {"forward": (1.0, 0.0), "steer_L": (0.0, 1.0), "steer_R": (0.0, -1.0), "stop": (0.0, 0.0)}
OBS_COLORS = ("red", "green", "blue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--repo-id", default="local/quaddle_l3")
    ap.add_argument("--root", default=str(HERE / "outputs/dataset_l3"))
    ap.add_argument("--seed", type=int, default=2000)
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
        color = cv.COLORS[ep % 3]
        layout = cv.random_layout(rng)
        tgt_xy = layout[cv.COLORS.index(color)]
        # obstacles = the other two blocks; ensure at least one lies roughly between start and target
        obs_xy = [layout[i] for i in range(3) if i != cv.COLORS.index(color)]
        th_t = math.atan2(tgt_xy[1], tgt_xy[0])
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
        frames = []; success = fell = collided = False; t = 0.0
        while t < cv.TIMEOUT_S:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            d_world = np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
            dist = float(np.linalg.norm(d_world[:2]))
            d_body = r.T @ d_world
            bearing = math.atan2(d_body[0], -d_body[1])
            # obstacle repulsion: nearest obstacle ahead
            avoid = 0.0
            for ox, oy in obs_xy:
                od = np.array([ox - pos[0], oy - pos[1], 0.0])
                odist = float(np.linalg.norm(od[:2]))
                ob = r.T @ od
                obear = math.atan2(ob[0], -ob[1])
                if odist < 0.28 and abs(obear) < math.radians(40):
                    avoid += (-1.0 if obear > 0 else 1.0) * (0.28 - odist) / 0.28
                if odist < 0.09:
                    collided = True
            if collided: break
            if dist < cv.SUCCESS_DIST:
                skill = "stop"; success = True
            elif abs(avoid) > 0.15:
                skill = "steer_L" if avoid > 0 else "steer_R"
            elif abs(bearing) > deadzone:
                skill = "steer_L" if bearing > 0 else "steer_R"
            else:
                skill = "forward"
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
                "task": f"walk to the {color} block avoiding obstacles",
            })
            if success: break
        if success and not collided and frames:
            for fr in frames: dataset.add_frame(fr)
            dataset.save_episode(); n_saved += 1
            if n_saved % 40 == 0: print(f"L3 saved {n_saved}/{args.episodes}", flush=True)
        else:
            n_fail += 1
    dataset.finalize()
    print(f"L3_DONE saved={n_saved} fail={n_fail} -> {ds_root}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()