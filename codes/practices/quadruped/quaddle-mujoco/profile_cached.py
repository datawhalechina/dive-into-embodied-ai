"""Stage-profile the cached training step reading from the NAS cache."""
import os, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler
from lerobot.datasets import EpisodeAwareSampler, LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.common.vla_utils import pad_vector
from cached_smolvla import CachedSmolVLA
from train_cached_v5 import CacheDataset

QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
CKPT = f"{QD}/outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model"
CACHE = "/dataset/mtl_data/quaddle_vla/cache_emb_v51"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda")
cfg = policy.config
cached = CachedSmolVLA(policy)
ds = CacheDataset(CACHE)
lang_emb_table = torch.from_numpy(ds.meta["lang_emb"]).view(torch.bfloat16).cuda()
lang_mask_table = torch.from_numpy(ds.meta["lang_mask"]).bool().cuda()
opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.95))
trainable = [p for p in policy.parameters() if p.requires_grad]
meta = LeRobotDatasetMetadata("local/quaddle_nav_352v51", root=f"{QD}/outputs/dataset_merged51")
sampler = EpisodeAwareSampler(meta.episodes["dataset_from_index"], meta.episodes["dataset_to_index"], shuffle=True, seed=1000)
dl = DataLoader(ds, batch_sampler=BatchSampler(sampler, 32, drop_last=True), num_workers=4, persistent_workers=True, pin_memory=True)
it = iter(dl)
policy.train()

times = {}
def tick(name, t0):
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times[name] = times.get(name, 0) + (t1 - t0)
    return t1

N = 20
for step in range(N):
    t = time.perf_counter()
    b = next(it)
    img = b["img"].cuda().view(torch.bfloat16)
    task = b["task"].cuda()
    lang_emb = lang_emb_table[task]; lang_mask = lang_mask_table[task]
    state = pad_vector(b["state"].cuda(), 32)
    action = pad_vector(b["action"].cuda(), 32)
    pad = b["pad"].cuda()
    t = tick("data+prep", t)
    kv, fpad = cached.frozen_prefill(img, lang_emb, lang_mask)
    t = tick("frozen_prefill", t)
    loss = cached.loss_from_batch(kv, fpad, state, action, pad)
    t = tick("grad_fwd+loss", t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    t = tick("backward", t)
    torch.nn.utils.clip_grad_norm_(trainable, 10.0)
    opt.step()
    t = tick("clip+optim", t)
    if step == 4:
        times.clear()

n = N - 5
print("=== cached step stage breakdown (ms) ===")
for k, v in times.items():
    print(f"{k:18s} {v/n*1000:7.1f}")
print(f"{'TOTAL':18s} {sum(times.values())/n*1000:7.1f}  ({32/(sum(times.values())/n):.0f} smp/s)")