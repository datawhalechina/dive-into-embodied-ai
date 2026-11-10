"""W5 评估：训练后的 SmolVLA 闭环 rollout，录制 GIF。

流程：加载正式检查点 → MuJoCo 闭环（VLA 推理动作块 → 阈值量化档位 →
查表 CPG 执行）→ overhead 相机录帧 → 拼接多任务 GIF。

推理节流：SmolVLA 输出 50 步动作块（10Hz×5s），每 0.5s 重推理一次，
块内按步推进——模拟真机异步推理形态。

用法（vla/act 环境）：
    MUJOCO_GL=egl uv run python ../quaddle-data/eval_rollout_gif.py \
        --ckpt outputs/smolvla_quaddle_v1/checkpoints/last/pretrained_model \
        --out outputs/rollout.gif --episodes 3
"""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CS123 = HERE.parent.parent / "quadruped" / "cs123"
sys.path.insert(0, str(CS123 / "8.rl_quaddle"))

import mujoco  # noqa: E402


def quantize(action2: np.ndarray) -> tuple[str, tuple[float, float]]:
    """连续动作 → 技能：argmax 判别档位类型，转向保留连续强度。"""
    vx, wz = float(action2[0]), float(action2[1])
    if max(abs(vx), abs(wz)) < 0.12:
        return ("stop", (0.0, 0.0))
    if abs(wz) >= abs(vx):
        return ("turn_left", (0.0, abs(wz))) if wz > 0 else ("turn_right", (0.0, -abs(wz)))
    return ("forward", (1.0, 0.0)) if vx > 0 else ("stop", (0.0, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(HERE / "outputs" / "smolvla_quaddle_v1" / "checkpoints" / "last" / "pretrained_model"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "rollout.gif"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--gif-fps", type=int, default=10)
    args = ap.parse_args()

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    import imageio.v2 as imageio
    from lerobot.datasets import LeRobotDataset  # noqa: F401  （确认环境）
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    ckpt = Path(args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device=device)

    # 场景：复用数据集的 scene.xml（含积木与三相机）
    scene_xml = HERE / "outputs" / "dataset" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    from cpg_controller import CpgController  # noqa: E402

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
    geom_ids = {
        c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
        for c in ("red", "green", "blue")
    }
    renderer = mujoco.Renderer(model, height=320, width=320)

    COLORS = ("red", "green", "blue")
    COLOR_RGBA = {"red": (0.9, 0.2, 0.2, 1), "green": (0.2, 0.8, 0.3, 1), "blue": (0.25, 0.4, 0.9, 1)}
    rng = np.random.default_rng(args.seed)

    def random_layout(rng):
        for _ in range(200):
            pts = []
            while len(pts) < 3:
                r = rng.uniform(0.25, 0.55)
                th = rng.uniform(-math.pi, math.pi)
                p = (r * math.cos(th), r * math.sin(th))
                if all(math.dist(p, q) >= 0.22 for q in pts):
                    pts.append(p)
            return pts
        raise RuntimeError

    def yaw_quat(yaw):
        return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    all_frames = []
    results = []
    for ep in range(args.episodes):
        color = COLORS[ep % 3]
        layout = random_layout(rng)
        target_xy = np.array(layout[COLORS.index(color)])
        yaw0 = float(rng.uniform(-math.pi, math.pi))

        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[3:7] = yaw_quat(yaw0)
        for i, c in enumerate(COLORS):
            model.geom_pos[geom_ids[c]][:2] = layout[i]
        mujoco.mj_forward(model, data)
        data.ctrl[:] = 0.0
        for _ in range(int(1.2 / model.opt.timestep)):
            mujoco.mj_step(model, data)

        ctrl = CpgController.__new__(CpgController)
        ctrl.env = types.SimpleNamespace(n_substeps=20)
        ctrl.model, ctrl.data = model, data
        ctrl.dt = 0.02
        ctrl._phases = np.zeros(4)
        ctrl._amp_scale, ctrl._dir = 0.0, 1.0
        ctrl.skill = "stop"

        ep_frames = []
        chunk = None
        chunk_i = 0
        action_log = []
        success, fell = False, False
        t = 0.0
        n_ctrl_per_gif_frame = int(50 // args.gif_fps)  # 每 GIF 帧 0.1s
        INFER_EVERY = 25  # 每 0.5s 重推理一次动作块

        while t < args.max_s:
            r = data.body(body_id).xmat.reshape(3, 3)
            pos = data.body(body_id).xpos
            dist = float(np.linalg.norm((target_xy - pos[:2])))

            if chunk is None or chunk_i >= INFER_EVERY:
                # 组装 VLA 观测
                imgs = {}
                for cam in ("overhead", "front", "oblique"):
                    renderer.update_scene(data, camera=cam)
                    imgs[cam] = renderer.render().copy()
                state = np.zeros(8, dtype=np.float32)
                joint_q = [data.qpos[7 + i * 6] for i in range(4)]
                state[:4] = joint_q
                v_body = r.T @ data.qvel[:3]
                state[4] = float(-v_body[1])
                state[5] = float((r.T @ data.qvel[3:6])[2])
                d_body = r.T @ np.array([target_xy[0] - pos[0], target_xy[1] - pos[1], 0.0])
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

            # select_action 返回单步动作 (1,2)——chunk 步进由 policy 内部管理，
            # 外部每 0.5s 重推理 = 滚动重规划（MPC 式）
            act = chunk[0].float().cpu().numpy()
            chunk_i += 1
            skill, act_vec = quantize(act)
            if skill.startswith("turn"):
                ctrl.set_turn_skill(skill.split("_")[1], abs(act_vec[1]))
            else:
                ctrl.set_skill(skill)
            action_log.append(skill)

            for _ in range(5):  # 0.1s
                ctrl.step()
                t += ctrl.dt
                if data.body(body_id).xpos[2] < 0.030:
                    fell = True
                    break
            if fell:
                break

            # 录 GIF 帧（每 0.1s 一帧）
            renderer.update_scene(data, camera="overhead")
            ep_frames.append(renderer.render().copy())

            if dist < 0.12:
                ctrl.set_skill("stop")
                for _ in range(10):
                    ctrl.step()
                    renderer.update_scene(data, camera="overhead")
                    ep_frames.append(renderer.render().copy())
                success = True
                break

        results.append((color, success, fell, t, dist))
        tag = "✓到达" if success else ("✗翻倒" if fell else "✗超时")
        print(f"ep{ep}: {color:5s} {tag} 用时 {t:.1f}s 末距离 {dist*100:.0f}cm "
              f"档位统计 { {s: action_log.count(s) for s in set(action_log)} }")
        # 拼 GIF：帧角落标注目标颜色
        all_frames.extend(ep_frames)

    # 合成单个 GIF（竖排标注省略——直接顺序拼接）
    imageio.mimsave(args.out, all_frames, fps=args.gif_fps, loop=0)
    print(f"GIF 已保存：{args.out}（{len(all_frames)} 帧，{args.gif_fps}fps）")
    n_ok = sum(1 for r in results if r[1])
    print(f"成功率: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
