"""诊断 4：训练集观测直接喂给模型，对照数据集动作（归一化/分布偏移裁决）。"""
import math
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

HERE = Path(__file__).resolve().parent
ds = LeRobotDataset("local/quaddle_nav_250v5",
                    root="/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_merged")
ckpt = str(HERE / "outputs" / "smolvla_quaddle_v5" / "checkpoints" / "last" / "pretrained_model")
cfg = PreTrainedConfig.from_pretrained(ckpt)
policy = SmolVLAPolicy.from_pretrained(ckpt)
policy.eval()
pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device="cuda")

print("数据集总帧数:", len(ds))
# 取若干训练帧：前进段、转向段各抽查
for idx in (100, 2000, 5000, 20000, 40000, 60000, 80000):
    fr = ds[idx]
    act = fr["action"].numpy()
    st = fr["observation.state"].numpy()
    batch = {
        "observation.images.camera1": fr["observation.images.camera1"].unsqueeze(0),
        "observation.images.camera2": fr["observation.images.camera2"].unsqueeze(0),
        "observation.images.camera3": fr["observation.images.camera3"].unsqueeze(0),
        "observation.state": torch.from_numpy(st).unsqueeze(0),
        "task": [fr["task"]],
    }
    batch = pre(batch)
    policy.reset()  # 关键：select_action 有内部动作队列，必须清空才能真正重推理
    with torch.no_grad():
        out = policy.select_action(batch)
    a = out[0].float().cpu().numpy()
    print(f"frame {idx:6d}  数据动作=({act[0]:+.1f},{act[1]:+.1f})  "
          f"bearing={math.degrees(st[6]):+6.1f}°  模型输出=({a[0]:+.2f},{a[1]:+.2f})")
os._exit(0)
