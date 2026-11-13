"""L1 instruction-following eval: command -> action consistency (target >=90%).

For each command (walk forward/turn left/turn right/stop), roll out the policy and
measure the fraction of frames whose quantized skill matches the commanded skill.
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import collect_v5 as cv
from eval_rollout_v5 import quantize

HERE = Path(__file__).resolve().parent
CMDS = [("walk forward", "forward"), ("turn left", "steer_L"),
        ("turn right", "steer_R"), ("stop", "stop")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch, mujoco
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt); policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device=device)
    model = mujoco.MjModel.from_xml_path(str(HERE / "outputs/dataset_w0/scene.xml"))
    model.opt.timestep = cv.DT_PHYS
    data = mujoco.MjData(model)
    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    renderer = mujoco.Renderer(model, height=cv.IMG, width=cv.IMG)
    rng = np.random.default_rng(args.seed)

    total_match = total_frames = 0
    for cmd, skill in CMDS:
        match = frames = 0
        for tr in range(args.trials):
            mujoco.mj_resetData(model, data)
            yaw0 = float(rng.uniform(-math.pi, math.pi))
            data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
            data.ctrl[:] = 0; mujoco.mj_forward(model, data)
            for _ in range(int(cv.SETTLE_S / cv.DT_PHYS)): mujoco.mj_step(model, data)
            amp_scale = phases_t = 0.0
            t = 0.0
            while t < args.window:
                r = data.body(body_id).xmat.reshape(3, 3)
                v_body = r.T @ data.qvel[:3]
                state = np.concatenate([
                    np.array([data.qpos[servo_qposadr[i]] for i in range(4)]),
                    [float(-v_body[1]), float((r.T @ data.qvel[3:6])[2])],
                    [0.0, 0.0]]).astype(np.float32)
                imgs = {}
                for cam in ("overhead", "front", "oblique"):
                    renderer.update_scene(data, camera=cam); imgs[cam] = renderer.render().copy()
                batch = {
                    "observation.images.camera1": torch.from_numpy(imgs["overhead"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.camera2": torch.from_numpy(imgs["front"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.camera3": torch.from_numpy(imgs["oblique"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.state": torch.from_numpy(state).unsqueeze(0),
                    "task": [cmd],
                }
                batch = pre(batch)
                with torch.no_grad():
                    act = policy.select_action(batch)[0].float().cpu().numpy()
                got, _ = quantize(act)
                # ignore ramp-up frames (first 1s) where amplitude still ramping
                if t > 1.0:
                    frames += 1
                    if got == skill: match += 1
                steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(got, 0.0)
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
        rate = match / frames if frames else 0
        total_match += match; total_frames += frames
        print(f"L1 {cmd:14s} -> {skill:8s} consistency {rate*100:.1f}% ({match}/{frames})", flush=True)
    overall = total_match / total_frames if total_frames else 0
    print(f"L1_OVERALL {overall*100:.1f}% (target>=90%)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()