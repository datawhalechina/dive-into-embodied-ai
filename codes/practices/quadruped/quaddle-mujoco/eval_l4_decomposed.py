"""L4 decomposed eval: hierarchical orchestrator + L2 nav skill (no L4 ckpt).

Breaks the L4 architecture ceiling (end-to-end 0/20) by task decomposition:
  (a) "stop when you see green": nav to green; emit stop when dist<0.20.
  (b) "go to red then blue": nav red (<12cm) -> nav blue (<12cm), in order.

The orchestrator re-issues the L2 skill's native training phrase
("walk to the {color} block") per stage and runs the same temporal-ensembling
executor as the 85% L2 eval (replan5, k=0.01). Stage switching uses the same
target-state interface the policy itself already consumes (state[6:8]
bearing/dist — on a real robot this maps to a target-color tracker).

Success criteria and RNG/scene protocol are byte-identical to eval_l4.py for
an apples-to-apples comparison against the end-to-end 0% baseline.
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
    ap.add_argument("--ens-k", type=float, default=0.01)
    ap.add_argument("--assist-deg", type=float, default=45.0,
                    help="task B turn-assist trigger (0=off); releases at 0.7x")
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--start-ep", type=int, default=0, help="fast-forward RNG to this episode (debug single ep)")
    ap.add_argument("--skill-log", action="store_true", help="log skill/dist/bearing every 10 frames")
    ap.add_argument("--mp4-out", default="", help="直出 960x720 H.264（高清重渲染）；设置后不再产 GIF")
    ap.add_argument("--mp4-size", default="960x720")
    ap.add_argument("--mp4-fps", type=int, default=25)
    ap.add_argument("--out", default=str(HERE / "outputs/rollout_l4_decomposed.gif"))
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
    model = mujoco.MjModel.from_xml_path(str(HERE / "outputs/dataset_l4/scene.xml"))
    model.opt.timestep = cv.DT_PHYS
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
    for ff in range(args.start_ep):   # 快进：复现逐集 RNG 消耗（layout + yaw0）
        cv.random_layout(rng); rng.uniform(-math.pi, math.pi)
    all_frames, results = [], []
    for ep in range(args.start_ep, args.episodes):
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
        chunks = []; frame = 0; force_replan = True; assist_active = False; cap_n = 0
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
            # ---- 编排器：阶段推进/停停触发（与策略 state[6:8] 同源的目标状态接口）----
            forced_stop = False
            if task_type == 0:
                if dist < 0.20: forced_stop = True
            else:
                if dist < cv.SUCCESS_DIST:
                    reached[stage] = True
                    if stage < len(targets) - 1:
                        print(f"  [ep{ep}] stage {targets[stage]} reached at t={t:.1f}s, switch -> {targets[stage+1]}", flush=True)
                        stage += 1
                        chunks = []; force_replan = True   # 新技能调用：清空集成缓冲
                        skill0 = None; last_switch_t = t
                    else:
                        success = True; break
            # ---- 转向辅助（仅 task B：|bearing|>assist_deg 即脚本转向，<0.7x 交还策略。
            #      贴前一目标起步+大方位角不在 L2 训练分布内，策略在此失锁且可能跑飞）----
            assist = False
            if task_type == 1 and args.assist_deg > 0 and assist_active:
                if abs(bearing) < math.radians(args.assist_deg * 0.7):
                    assist_active = False; force_replan = True
                else:
                    assist = True
                    skill_now = "steer_L" if bearing > 0 else "steer_R"
            elif task_type == 1 and args.assist_deg > 0 and abs(bearing) > math.radians(args.assist_deg):
                assist_active = True; assist = True
                skill_now = "steer_L" if bearing > 0 else "steer_R"
                chunks = []; skill0 = None; last_switch_t = t
            # ---- L2 技能（原生短语 + temporal ensembling 执行器）----
            if not assist and (force_replan or frame % args.replan_steps == 0):
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
                    "task": [f"walk to the {targets[stage]} block"],
                }
                batch = pre(batch)
                with torch.no_grad():
                    ch = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                chunks.append((frame, ch))
                chunks = chunks[-12:]
                force_replan = False
            if assist:
                skill = skill_now
            else:
                num = np.zeros(2); den = 0.0
                for s, a in chunks:
                    idx = frame - s
                    if 0 <= idx < len(a):
                        w = math.exp(-args.ens_k * idx)
                        num += w * a[idx]; den += w
                act = num / den if den > 0 else np.zeros(2)
                skill, _ = quantize(act)
                if forced_stop: skill = "stop"
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
            if args.skill_log and frame % 10 == 0:
                print(f"  [ep{ep}] t={t:5.1f} stage={targets[stage]} skill={skill:7s} dist={dist*100:5.1f}cm bearing={math.degrees(bearing):6.1f}deg assist={int(assist or assist_active)} nchunk={len(chunks)}", flush=True)
            for _ in range(5):
                for _ in range(cv.SUB): mujoco.mj_step(model, data)
                t += cv.DT_CTRL
                if data.body(body_id).xpos[2] < 0.030: fell = True; break
                if mp4_proc is not None:
                    cap_n += 1
                    if cap_n % 2 == 0:  # 每 2 个控制步 1 帧（0.04s）@25fps = 1x 真实速度
                        gif_renderer.update_scene(data, camera=gif_cam)
                        mp4_proc.stdin.write(gif_renderer.render().tobytes())
                else:
                    renderer.update_scene(data, camera=gif_cam); ep_frames.append(renderer.render().copy())
            if fell: break
            frame += 1
            if task_type == 0 and forced_stop: success = True; break  # dist<0.20 且本步发 stop（镜像 eval_l4 判据）
        results.append(success)
        print(f"ep{ep}: {lang} {'✓' if success else '✗'} reached={reached} fell={fell} t={t:.1f}s stage={targets[stage]} dist={dist*100:.0f}cm", flush=True)
        all_frames.extend(ep_frames)
    if mp4_proc is not None:
        mp4_proc.stdin.close(); mp4_proc.wait()
        print(f"MP4_SAVED {args.mp4_out}", flush=True)
    else:
        imageio.mimsave(args.out, all_frames, fps=10, loop=0)
    n_ok = sum(results)
    print(f"L4_SUCCESS {n_ok}/{len(results)} (end-to-end baseline 0/20)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
