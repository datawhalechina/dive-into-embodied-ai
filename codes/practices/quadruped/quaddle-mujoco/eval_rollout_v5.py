"""v5 评估：SmolVLA 闭环 rollout（quaddle-mujoco 环境，2 维动作 steer 执行器）。

动作 → 技能量化（沿用 v3 的 argmax 相对判别，执行器改为可靠的 duty 转向）：
    max(|vx|,|wz|) < 0.12          → stop
    |wz| ≥ |vx| 且 |wz| > 0.15     → steer_L/R（wz 符号 = 方向，全幅 duty 不对称）
    否则                            → forward（vx > 0）或 stop

用法：
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib \
      HF_HUB_OFFLINE=1 HF_HOME=/workspace/.cache/huggingface python eval_rollout_v5.py \
      --ckpt outputs/smolvla_quaddle_v5/checkpoints/last/pretrained_model \
      --episodes 3 --seed 7000 --out outputs/rollout_v5.gif
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XML = HERE / "models" / "scene.xml"
CONFIG = json.loads((HERE / "gait.json").read_text())

DT_PHYS, DT_CTRL = 0.001, 0.02
SUB = 20
SETTLE_S = float(CONFIG.get("settle_seconds", 2.0))
RAMP_S = float(CONFIG.get("ramp_seconds", 1.5))
CTRL_MAX = 1.2
SIGNS = np.array(CONFIG["actuator_signs"])
BIAS = np.array(CONFIG["bias_rad"])
AMPS = np.array(CONFIG["amplitude_rad"])
PHASE0 = np.array(CONFIG["phase_rad"])
FREQ = float(CONFIG["frequency_hz"])
DUTY = float(CONFIG["sweep_fraction"])
STEER_LIMIT = float(CONFIG["steering_limit"])
SIDE = {"L": (0, 3), "R": (1, 2)}
DWELL_S = 0.4

COLORS = ("red", "green", "blue")


def wave_at(ph, duty):
    cycle = (ph / (2 * math.pi)) % 1.0
    phased = np.where(cycle < duty, math.pi * cycle / duty,
                      math.pi + math.pi * (cycle - duty) / (1 - duty))
    return np.cos(phased)


def quantize(action2: np.ndarray):
    """连续动作 → 技能。支持 2 维 [vx,wz]（v5）与 1 维 mode（#3）。
    2 维：v5 诊断后互斥硬约束（vx<0 置零、|wz|>0.5 强制纯转向）+ argmax 相对判别。
    1 维：mode>0.5→steer_L，mode<-0.5→steer_R，否则 forward（stop 并入前进，dist<12cm 判停）。"""
    if len(action2) == 1:
        mode = float(action2[0])
        if mode > 0.5:
            return ("steer_L", (0.0, 1.0))
        if mode < -0.5:
            return ("steer_R", (0.0, -1.0))
        return ("forward", (1.0, 0.0))
    vx, wz = float(action2[0]), float(action2[1])
    if vx < 0:
        vx = 0.0
    if abs(wz) > 0.5:
        return ("steer_L", (0.0, 1.0)) if wz > 0 else ("steer_R", (0.0, -1.0))
    if max(abs(vx), abs(wz)) < 0.12:
        return ("stop", (0.0, 0.0))
    if abs(wz) >= abs(vx) and abs(wz) > 0.15:
        return ("steer_L", (0.0, 1.0)) if wz > 0 else ("steer_R", (0.0, -1.0))
    return ("forward", (1.0, 0.0)) if vx > 0 else ("stop", (0.0, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(HERE / "outputs" / "smolvla_quaddle_v5" / "checkpoints" / "last" / "pretrained_model"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "rollout_v5.gif"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-s", type=float, default=60.0, help="单回合上限（本环境步速慢，放宽到 60s）")
    ap.add_argument("--fm-steps", type=int, default=0)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--replan-steps", type=int, default=5,
                    help="每执行多少步重新规划一次（默认5=0.5s，打破自旋承诺、走路更顺；25=2.5s 为旧基线）")
    args = ap.parse_args()

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    import imageio.v2 as imageio
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    import mujoco

    ckpt = Path(args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.eval()
    if args.fm_steps > 0:
        fm_attr = next((a for a in ("num_steps", "n_steps", "num_inference_steps")
                        if hasattr(policy.config, a)), None)
        if fm_attr:
            setattr(policy.config, fm_attr, args.fm_steps)
            print(f"flow matching 步数: {fm_attr} = {args.fm_steps}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device=device)

    scene_xml = HERE / "outputs" / "dataset_w0" / "scene.xml"  # v5 场景在各 worker 目录（合并只含 lerobot 数据）
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.timestep = DT_PHYS
    data = mujoco.MjData(model)

    servo_qposadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
                for c in ("red", "green", "blue")}
    renderer = mujoco.Renderer(model, height=320, width=320)
    # GIF 视角：自由相机对准场地中心的 3/4 俯瞰（场景 named "oblique" 相机光轴背离场地，
    # 渲染静态背景，故不用它）；机器人移动/转向清晰可见
    gif_cam = mujoco.MjvCamera()
    gif_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    gif_cam.lookat[:] = [0, 0, 0]
    gif_cam.distance = 1.3
    gif_cam.azimuth = 135
    gif_cam.elevation = -35

    rng = np.random.default_rng(args.seed)

    def random_layout(rng):
        pts = []
        while len(pts) < 3:
            r = rng.uniform(0.25, 0.55)
            th = rng.uniform(-math.pi, math.pi)
            p = (r * math.cos(th), r * math.sin(th))
            if all(math.dist(p, q) >= 0.22 for q in pts):
                pts.append(p)
        return pts

    all_frames, results = [], []
    for ep in range(args.episodes):
        color = COLORS[ep % 3]
        layout = random_layout(rng)
        tgt_xy = np.array(layout[COLORS.index(color)])
        th_t = math.atan2(tgt_xy[1], tgt_xy[0])
        yaw0 = float(rng.uniform(-math.pi, math.pi))

        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
        for i, c in enumerate(("red", "green", "blue")):
            model.geom_pos[geom_ids[c]][:2] = list(layout[i])
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(int(SETTLE_S / DT_PHYS)):
            mujoco.mj_step(model, data)

        amp_scale, phases_t = 0.0, 0.0
        chunk, chunk_i = None, 0
        skill0, last_switch_t = None, 0.0
        action_log = []
        success, fell, t = False, False, 0.0
        ep_frames = []

        while t < args.max_s:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            dist = float(np.linalg.norm(tgt_xy - pos[:2]))

            if chunk is None or chunk_i >= args.replan_steps:
                imgs = {}
                for cam in ("overhead", "front", "oblique"):
                    renderer.update_scene(data, camera=cam)
                    imgs[cam] = renderer.render().copy()
                state = np.zeros(8, dtype=np.float32)
                state[:4] = [data.qpos[servo_qposadr[i]] for i in range(4)]
                v_body = r.T @ data.qvel[:3]
                state[4] = float(-v_body[1])
                state[5] = float((r.T @ data.qvel[3:6])[2])
                d_body = r.T @ np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
                state[6] = math.atan2(d_body[0], -d_body[1])
                state[7] = dist
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

            act = chunk[0].float().cpu().numpy()
            chunk_i += 1
            skill, _ = quantize(act)
            if skill != skill0 and skill0 is not None and t - last_switch_t < DWELL_S and skill != "stop":
                skill = skill0

            steer_now = {"steer_L": 1.0, "steer_R": -1.0}.get(skill, 0.0)
            amp_scale = min(1.0, amp_scale + DT_CTRL / RAMP_S)
            phases_t += 2 * math.pi * FREQ * DT_CTRL
            duty = np.full(4, DUTY)
            if steer_now > 0:
                duty[list(SIDE["L"])] *= (1 - steer_now * STEER_LIMIT)
                duty[list(SIDE["R"])] *= (1 + steer_now * STEER_LIMIT)
            elif steer_now < 0:
                duty[list(SIDE["L"])] *= (1 + abs(steer_now) * STEER_LIMIT)
                duty[list(SIDE["R"])] *= (1 - abs(steer_now) * STEER_LIMIT)
            w = wave_at(phases_t + PHASE0, duty)
            data.ctrl[:] = np.clip(SIGNS * (BIAS + AMPS * amp_scale * w), -CTRL_MAX, CTRL_MAX)
            if skill != skill0:
                skill0, last_switch_t = skill, t
            action_log.append(skill)

            for _ in range(5):
                for _ in range(SUB):
                    mujoco.mj_step(model, data)
                t += DT_CTRL
                if data.body(body_id).xpos[2] < 0.030:
                    fell = True
                    break
            if fell:
                break

            renderer.update_scene(data, camera=gif_cam)
            ep_frames.append(renderer.render().copy())
            if dist < 0.12:
                success = True
                break

        results.append((color, success, fell, t, dist))
        tag = "✓到达" if success else ("✗翻倒" if fell else "✗超时")
        print(f"ep{ep}: {color:5s} {tag} 用时 {t:.1f}s 末距离 {dist*100:.0f}cm "
              f"档位统计 { {s: action_log.count(s) for s in set(action_log)} }", flush=True)
        all_frames.extend(ep_frames)

    imageio.mimsave(args.out, all_frames, fps=args.gif_fps, loop=0)
    n_ok = sum(1 for r in results if r[1])
    print(f"GIF 已保存：{args.out}（{len(all_frames)} 帧）")
    print(f"成功率: {n_ok}/{len(results)}")
    os._exit(0)


if __name__ == "__main__":
    main()
