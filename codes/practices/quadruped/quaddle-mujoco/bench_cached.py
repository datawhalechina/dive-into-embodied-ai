"""Micro-benchmark the cached per-step compute (vision excluded, as the cache eliminates it).
Also tests torch.compile and a few batch sizes. Validates the trainer compute path."""
import os, time, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
from torch.utils.data import DataLoader
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from cached_smolvla import CachedSmolVLA

QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
CKPT = f"{QD}/outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model"
MERGED = f"{QD}/outputs/dataset_merged51"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda")
cfg = policy.config; m = policy.model; vlm = m.vlm_with_expert
cached = CachedSmolVLA(policy)
meta = LeRobotDatasetMetadata("local/quaddle_nav_352v51", root=MERGED)
ds = LeRobotDataset("local/quaddle_nav_352v51", root=MERGED,
                    delta_timestamps=resolve_delta_timestamps(cfg, meta), return_uint8=True)
pre, _ = make_smolvla_pre_post_processors(cfg, ds.meta.stats)
dim = vlm.config.text_config.hidden_size
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.95))
trainable = [p for p in policy.parameters() if p.requires_grad]


def make_inputs(B):
    dl = DataLoader(ds, batch_size=B, shuffle=True, num_workers=4, drop_last=True)
    b = next(iter(dl))
    for k in b:
        if "images" in k and b[k].dtype == torch.uint8:
            b[k] = b[k].to(torch.float32) / 255.0
    b = pre(b)
    images, _ = policy.prepare_images(b)
    embs = []
    with torch.no_grad():
        for img in images:
            ie = vlm.embed_image(img) * math.sqrt(dim)
            embs.append(ie)
        img_emb = torch.cat(embs, 1)
        lang_emb = vlm.embed_language_tokens(b[OBS_LANGUAGE_TOKENS]) * math.sqrt(dim)
    lang_mask = b[OBS_LANGUAGE_ATTENTION_MASK]
    state = pad_vector(b[OBS_STATE][:, -1, :].cuda(), 32)
    action = pad_vector(b[ACTION].cuda(), 32)
    pad = b["action_is_pad"].cuda()
    return img_emb, lang_emb, lang_mask, state, action, pad


def bench(B, compile_mode=None, n=20):
    c = cached
    if compile_mode:
        c = CachedSmolVLA(policy)
        c.frozen_prefill = torch.compile(c.frozen_prefill, mode=compile_mode)
        c.train_forward = torch.compile(c.train_forward, mode=compile_mode)
    img_emb, lang_emb, lang_mask, state, action, pad = make_inputs(B)
    policy.train()

    def step():
        kv, fpad = c.frozen_prefill(img_emb, lang_emb, lang_mask)
        loss = c.loss_from_batch(kv, fpad, state, action, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        opt.step()
        return loss

    for _ in range(5):
        loss = step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        loss = step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / n * 1000
    print(f"B={B:3d} compile={str(compile_mode):28s} {ms:7.1f} ms/step  {B/(ms/1000):7.1f} smp/s  "
          f"loss={loss.item():.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
    return ms


print("=== cached per-step compute (vision excluded) ===", flush=True)
base = bench(32, None)
bench(32, "reduce-overhead")
bench(64, None)
bench(128, None)
print(f"original baseline ~509 ms/step (62.9 smp/s); cached B32 speedup ~ {509/base:.1f}x", flush=True)