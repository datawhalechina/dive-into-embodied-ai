"""L1 instruction-following collection: language command -> skill (no navigation).

Commands: walk forward / turn left / turn right / stop. The expert executes the
commanded skill for a fixed window; record (obs, language=command, action=skill).
Trains a language->action mapping (L1, target >=90% command-action consistency).
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import collect_v5 as cv

HERE = Path(__file__).resolve().parent
CMDS = [("walk forward", "forward"), ("turn left", "steer_L"),
        ("turn right", "steer_R"), ("stop", "stop")]
ACTIONS = {"forward": (1.0, 0.0), "steer_L": (0.0, 1.0), "steer_R": (0.0, -1.0), "stop": (0.0, 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--repo-id", default="local/quaddle_l1")
    ap.add_argument("--root", default=str(HERE / "outputs/dataset_l1"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--window", type=float, default=4.0)
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
    n = 0
    for ep in range(args.episodes):
        cmd, skill = CMDS[ep % 4]
        mujoco.mj_resetData(model, data)
        yaw0 = float(rng.uniform(-math.pi, math.pi))
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        data.ctrl[:] = 0; mujoco.mj_forward(model, data)
        for _ in range(int(cv.SETTLE_S / cv.DT_PHYS)): mujoco.mj_step(model, data)
        amp_scale = phases_t = 0.0
        steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
        frames = []
        t = 0.0
        while t < args.window:
            amp_scale = min(1.0, amp_scale + cv.DT_CTRL / cv.RAMP_S)
            phases_t += 2 * math.pi * cv.FREQ * cv.DT_CTRL
            duty = np.full(4, cv.DUTY)
            if steer_now > 0:
                duty[list(cv.SIDE["L"])] *= (1 - steer_now * cv.STEER_LIMIT); duty[list(cv.SIDE["R"])] *= (1 + steer_now * cv.STEER_LIMIT)
            elif steer_now < 0:
                duty[list(cv.SIDE["L"])] *= (1 + abs(steer_now) * cv.STEER_LIMIT); duty[list(cv.SIDE["R"])] *= (1 - abs(steer_now) * cv.STEER_LIMIT)
            w = cv.wave_at(phases_t + cv.PHASE0, duty)
            data.ctrl[:] = np.clip(cv.SIGNS * (cv.BIAS + cv.AMPS * amp_scale * w), -cv.CTRL_MAX, cv.CTRL_MAX)
            for _ in range(5):
                for _ in range(cv.SUB): mujoco.mj_step(model, data)
                t += cv.DT_CTRL
            r = data.body(body_id).xmat.reshape(3, 3)
            v_body = r.T @ data.qvel[:3]
            state = np.concatenate([
                np.array([data.qpos[servo_qposadr[i]] for i in range(4)]),
                [float(-v_body[1]), float((r.T @ data.qvel[3:6])[2])],
                [0.0, 0.0]]).astype(np.float32)
            frames.append({
                "observation.images.camera1": cv.renderer_update(model, data, renderer, "overhead"),
                "observation.images.camera2": cv.renderer_update(model, data, renderer, "front"),
                "observation.images.camera3": cv.renderer_update(model, data, renderer, "oblique"),
                "observation.state": state,
                "action": np.array(ACTIONS[skill], dtype=np.float32),
                "task": cmd,
            })
        for fr in frames: dataset.add_frame(fr)
        dataset.save_episode(); n += 1
        if n % 40 == 0: print(f"L1 saved {n}/{args.episodes}", flush=True)
    dataset.finalize()
    print(f"L1_DONE saved={n} -> {ds_root}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()