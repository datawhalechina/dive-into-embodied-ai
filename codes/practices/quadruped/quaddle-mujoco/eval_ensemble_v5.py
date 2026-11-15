"""Temporal-ensembling eval (ACT-style) to attack OOD self-spin, VLA no-RL.

Instead of draining a fixed action queue, replan via predict_action_chunk every
--replan-steps frames and execute a recency-weighted average of overlapping chunk
predictions (temporal ensembling). Smoother, more stable actions -> less spin lock.
"""
from __future__ import annotations
import argparse, math, os, subprocess
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
    ap.add_argument("--ens-k", type=float, default=0.01, help="recency weight exp(-k*age)")
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--out", default=str(HERE / "outputs/rollout_ensemble.gif"))
    ap.add_argument("--gif-fps", type=int, default=50, help="render every control step, play at this fps (silky)")
    ap.add_argument("--mp4-out", default="", help="直出 960x720 H.264（高清重渲染）；设置后不再产 GIF")
    ap.add_argument("--mp4-size", default="960x720")
    ap.add_argument("--mp4-fps", type=int, default=25)
    ap.add_argument("--gait-freq", type=float, default=0.0, help="override CPG freq (0=use gait.json)")
    ap.add_argument("--gait-amp-scale", type=float, default=1.0, help="scale leg amplitude (speed)")
    ap.add_argument("--servo-kp", type=float, default=0.0, help="override servo kp (0=scene.xml default 0.15)")
    ap.add_argument("--servo-kv", type=float, default=0.0, help="override servo kv (0=default 0.006)")
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
    torch.manual_seed(7000)  # flow 采样噪声种子化：策略输出确定可复现
    FREQ = args.gait_freq if args.gait_freq > 0 else cv.FREQ
    AMPS = cv.AMPS * args.gait_amp_scale

    model = mujoco.MjModel.from_xml_path(str(HERE / "outputs/dataset_w0/scene.xml"))
    model.opt.timestep = cv.DT_PHYS
    if args.servo_kp > 0:
        model.actuator_gainprm[:, 0] = args.servo_kp
        model.actuator_biasprm[:, 1] = -args.servo_kp
        model.actuator_forcerange[:] = [-max(0.245, args.servo_kp * 0.5), max(0.245, args.servo_kp * 0.5)]
    if args.servo_kv > 0:
        model.actuator_biasprm[:, 2] = -args.servo_kv
    data = mujoco.MjData(model)
    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}") for c in cv.COLORS}
    renderer = mujoco.Renderer(model, height=cv.IMG, width=cv.IMG)
    gif_cam = mujoco.MjvCamera(); gif_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    gif_cam.lookat[:] = [0, 0, 0]; gif_cam.distance = 1.3; gif_cam.azimuth = 135; gif_cam.elevation = -35
    # 高清 mp4：独立渲染器（策略观测仍 320²），帧流式喂 ffmpeg，不落巨型 GIF
    mp4_proc = None; GIF_W = GIF_H = cv.IMG
    if args.mp4_out:
        GIF_W, GIF_H = map(int, args.mp4_size.split("x"))
        ff = "/workspace/gifopt/ffbin/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2"
        mp4_proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{GIF_W}x{GIF_H}", "-r", str(args.mp4_fps), "-i", "-",
                                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                     "-pix_fmt", "yuv420p", args.mp4_out], stdin=subprocess.PIPE)
        gif_renderer = mujoco.Renderer(model, height=GIF_H, width=GIF_W)
        gif_cam.distance = 1.6

    rng = np.random.default_rng(args.seed)
    cap_n = 0  # 控制步抓帧计数（每 2 步 1 帧 → 0.04s/帧 @50fps = 2x 速丝滑）
    all_frames, results = [], []
    for ep in range(args.episodes):
        color = cv.COLORS[ep % 3]
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
        chunks = []  # (start_frame, actions(50,2))
        frame = 0
        success = fell = False; t = 0.0
        action_log = []
        ep_frames = []
        while t < args.max_s:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            dist = float(np.linalg.norm(tgt_xy - pos[:2]))
            d_body = r.T @ np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
            bearing = math.atan2(d_body[0], -d_body[1])
            if frame % args.replan_steps == 0:
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
            # temporal ensemble at current frame
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
            phases_t += 2 * math.pi * FREQ * cv.DT_CTRL
            duty = np.full(4, cv.DUTY)
            if steer_now > 0:
                duty[list(cv.SIDE["L"])] *= (1 - steer_now * cv.STEER_LIMIT); duty[list(cv.SIDE["R"])] *= (1 + steer_now * cv.STEER_LIMIT)
            elif steer_now < 0:
                duty[list(cv.SIDE["L"])] *= (1 + abs(steer_now) * cv.STEER_LIMIT); duty[list(cv.SIDE["R"])] *= (1 - abs(steer_now) * cv.STEER_LIMIT)
            w = cv.wave_at(phases_t + cv.PHASE0, duty)
            data.ctrl[:] = np.clip(cv.SIGNS * (cv.BIAS + AMPS * amp_scale * w), -cv.CTRL_MAX, cv.CTRL_MAX)
            if skill != skill0: skill0, last_switch_t = skill, t
            action_log.append(skill)
            for _ in range(5):
                for _ in range(cv.SUB): mujoco.mj_step(model, data)
                t += cv.DT_CTRL
                if data.body(body_id).xpos[2] < 0.030: fell = True; break
                if mp4_proc is not None:
                    cap_n += 1
                    if cap_n % 2 == 0:  # 每个控制步都抓帧判定：修复旧 %6 在 5 子步循环里"连抓 5 帧+静默 0.6s"的突跳
                        gif_renderer.update_scene(data, camera=gif_cam)
                        mp4_proc.stdin.write(gif_renderer.render().tobytes())
                else:
                    renderer.update_scene(data, camera=gif_cam); ep_frames.append(renderer.render().copy())
            if fell: break
            frame += 1
            if dist < cv.SUCCESS_DIST: success = True; break
        results.append((color, success, fell, t, dist))
        tag = "✓到达" if success else ("✗翻倒" if fell else "✗超时")
        print(f"ep{ep}: {color:5s} {tag} 用时{t:.1f}s 末距离{dist*100:.0f}cm 档位{ {s: action_log.count(s) for s in set(action_log)} }", flush=True)
        all_frames.extend(ep_frames)
    if mp4_proc is not None:
        mp4_proc.stdin.close(); mp4_proc.wait()
        print(f"MP4_SAVED {args.mp4_out}", flush=True)
    else:
        imageio.mimsave(args.out, all_frames, fps=args.gif_fps, loop=0)
    n_ok = sum(1 for r in results if r[1])
    print(f"成功率: {n_ok}/{len(results)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()