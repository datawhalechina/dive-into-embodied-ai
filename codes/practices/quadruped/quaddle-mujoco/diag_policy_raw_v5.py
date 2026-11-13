"""诊断 2：模型原始输出分布（ep0 初始观测，fm10/fm20 各 10 次采样）。"""
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

import mujoco

scene_xml = HERE / "outputs" / "dataset_w0" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene_xml))
model.opt.timestep = 0.001
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")
geom_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
            for c in ("red", "green", "blue")}

rng = np.random.default_rng(7000)  # 复现 eval ep0 的场景抽取
for ep in range(1):
    pts = []
    while len(pts) < 3:
        r = rng.uniform(0.25, 0.55)
        th = rng.uniform(-math.pi, math.pi)
        p = (r * math.cos(th), r * math.sin(th))
        if all(math.dist(p, q) >= 0.22 for q in pts):
            pts.append(p)
    yaw0 = float(rng.uniform(-math.pi, math.pi))
layout = pts
color = "red"
tgt_xy = np.array(layout[0])
mujoco.mj_resetData(model, data)
data.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
for i, c in enumerate(("red", "green", "blue")):
    model.geom_pos[geom_ids[c]][:2] = list(layout[i])
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
for _ in range(int(2.0 / 0.001)):
    mujoco.mj_step(model, data)

ckpt = str(HERE / "outputs" / "smolvla_quaddle_v5" / "checkpoints" / "last" / "pretrained_model")
cfg = PreTrainedConfig.from_pretrained(ckpt)
policy = SmolVLAPolicy.from_pretrained(ckpt)
policy.eval()
pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device="cuda")

renderer = mujoco.Renderer(model, height=320, width=320)


def obs_batch():
    imgs = {}
    for cam in ("overhead", "front", "oblique"):
        renderer.update_scene(data, camera=cam)
        imgs[cam] = renderer.render().copy()
    r = data.body(body_id).xmat.reshape(3, 3)
    pos = data.body(body_id).xpos
    state = np.zeros(8, dtype=np.float32)
    state[4:6] = [0.0, 0.0]
    d_body = r.T @ np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1], 0.0])
    state[6] = math.atan2(d_body[0], -d_body[1])
    state[7] = float(np.linalg.norm(np.array([tgt_xy[0] - pos[0], tgt_xy[1] - pos[1]])))
    return {
        "observation.images.camera1": torch.from_numpy(imgs["overhead"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
        "observation.images.camera2": torch.from_numpy(imgs["front"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
        "observation.images.camera3": torch.from_numpy(imgs["oblique"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": [f"walk to the {color} block"],
    }


r = data.body(body_id).xmat.reshape(3, 3)
pos0 = data.body(body_id).xpos
d_vec = np.array([tgt_xy[0] - pos0[0], tgt_xy[1] - pos0[1], 0.0])
d_body = r.T @ d_vec
print(f"ep0 场景复现: bearing={math.degrees(math.atan2(d_body[0], -d_body[1])):.1f}° "
      f"dist={np.linalg.norm(d_vec[:2])*100:.0f}cm")

for fm in (10, 20):
    attr = next((a for a in ("num_steps",) if hasattr(policy.config, a)), None)
    if attr:
        setattr(policy.config, attr, fm)
    outs_vx, outs_wz = [], []
    for trial in range(10):
        batch = pre(obs_batch())
        with torch.no_grad():
            out = policy.select_action(batch)
        a = out[0].float().cpu().numpy()
        outs_vx.append(float(a[0]))
        outs_wz.append(float(a[1]))
    print(f"\nfm_steps={fm}: 10 次采样")
    print(f"  vx: mean={np.mean(outs_vx):+.3f} std={np.std(outs_vx):.3f}  [{min(outs_vx):+.3f},{max(outs_vx):+.3f}]")
    print(f"  wz: mean={np.mean(outs_wz):+.3f} std={np.std(outs_wz):.3f}  [{min(outs_wz):+.3f},{max(outs_wz):+.3f}]")
    print(f"  vx 样本: {[f'{v:+.2f}' for v in outs_vx]}")
    print(f"  wz 样本: {[f'{v:+.2f}' for v in outs_wz]}")
os._exit(0)
