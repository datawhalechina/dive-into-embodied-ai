"""诊断 5：预处理管线抽查（归一化是否 NaN/塌缩）+ stats 文件对比。"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

HERE = Path(__file__).resolve().parent
QD = HERE.parent.parent / "quaddle-mujoco" if False else HERE
MERGED = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_merged"
CKPT = str(HERE / "outputs" / "smolvla_quaddle_v5" / "checkpoints" / "last" / "pretrained_model")

ds = LeRobotDataset("local/quaddle_nav_250v5", root=MERGED)
fr = ds[5000]

cfg = PreTrainedConfig.from_pretrained(CKPT)
policy = SmolVLAPolicy.from_pretrained(CKPT)
policy.eval()
pre, post = make_pre_post_processors(cfg, policy_cfg_type="smolvla", device="cuda")

batch = {
    "observation.images.camera1": fr["observation.images.camera1"].unsqueeze(0),
    "observation.images.camera2": fr["observation.images.camera2"].unsqueeze(0),
    "observation.images.camera3": fr["observation.images.camera3"].unsqueeze(0),
    "observation.state": torch.from_numpy(fr["observation.state"].numpy()).unsqueeze(0),
    "task": [fr["task"]],
}
pb = pre(dict(batch))
print("== 预处理后的 state ==")
print(pb["observation.state"])
print("== 预处理后的 image1 ==")
img = pb["observation.images.camera1"]
print("dtype", img.dtype, "shape", tuple(img.shape), "mean", float(img.float().mean()), "std", float(img.float().std()), "nan", bool(torch.isnan(img).any()))
print("== policy config normalization (input_features) ==")
nrm = getattr(cfg, "input_features", None)
for k, v in (nrm or {}).items():
    print(" ", k, type(v).__name__, getattr(v, "stats", None) if not hasattr(v, "normalize") else {a: getattr(getattr(v, a, None), "to", lambda x=None: v) for a in []})

# checkpoint 内嵌 stats
import glob
for p in glob.glob(CKPT + "/*.json"):
    j = json.load(open(p))
    s = json.dumps(j)
    if "stat" in s.lower() and ("mean" in s):
        print("== ckpt 内含 stats 的文件:", os.path.basename(p), "size", len(s))
        d = j.get("stats") or j
        if isinstance(d, dict) and "observation.state" in json.dumps(d)[:20000]:
            sub = d.get("stats", d)
            print("  state stats:", json.dumps(sub.get("observation.state", {}))[:200])
print("== merged stats.json state 段 ==")
ms = json.load(open(MERGED + "/meta/stats.json"))
print("  state stats:", json.dumps(ms.get("observation.state", {}))[:300])
w0 = json.load(open("/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_w0/lerobot/meta/stats.json"))
print("== w0 stats.json state 段 ==")
print("  state stats:", json.dumps(w0.get("observation.state", {}))[:300])
os._exit(0)
