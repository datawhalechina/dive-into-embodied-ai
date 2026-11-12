"""Equivalence + speed benchmark: original SmolVLA forward vs cached split-grad forward."""
import os, time, math, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from cached_smolvla import CachedSmolVLA

QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
CKPT = f"{QD}/outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model"
MERGED = f"{QD}/outputs/dataset_merged51"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda")
cfg = policy.config
m = policy.model
vlm = m.vlm_with_expert
cached = CachedSmolVLA(policy)

meta = LeRobotDatasetMetadata("local/quaddle_nav_352v51", root=MERGED)
ds = LeRobotDataset("local/quaddle_nav_352v51", root=MERGED,
                    delta_timestamps=resolve_delta_timestamps(cfg, meta), return_uint8=True)
pre, post = make_smolvla_pre_post_processors(cfg, ds.meta.stats)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, drop_last=True)
batch = next(iter(dl))
for k in batch:
    if "images" in k and batch[k].dtype == torch.uint8:
        batch[k] = batch[k].to(torch.float32) / 255.0
batch = pre(batch)


def to_float_img(b):
    return b


def cached_inputs(batch):
    images, img_masks = policy.prepare_images(batch)
    dim = vlm.config.text_config.hidden_size
    img_embs = []
    for img in images:
        ie = vlm.embed_image(img)
        ie = ie * torch.tensor(dim ** 0.5, dtype=ie.dtype, device=ie.device)
        img_embs.append(ie)
    img_emb = torch.cat(img_embs, dim=1)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    lang_emb = vlm.embed_language_tokens(lang_tokens) * math.sqrt(dim)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    return img_emb, lang_emb, lang_mask, state, actions


# ---- equivalence ----
B = 8
actions = policy.prepare_action(batch)
noise = m.sample_noise(actions.shape, "cuda")
time_t = m.sample_time(B, "cuda")

policy.eval()
with torch.no_grad():
    loss_orig, _ = policy.forward(batch, noise=noise, time=time_t)
    img_emb, lang_emb, lang_mask, state, actions = cached_inputs(batch)
    kv, fpad = cached.frozen_prefill(img_emb, lang_emb, lang_mask)
    losses_c = cached.train_forward(kv, fpad, state, actions, noise=noise, time=time_t)
    losses_c = losses_c[:, :, :2]
    aip = batch.get("action_is_pad")
    if aip is not None:
        losses_c = losses_c * (~aip).unsqueeze(-1)
        loss_c = losses_c.sum() / ((~aip).sum() * losses_c.shape[-1]).clamp_min(1)
    else:
        loss_c = losses_c.mean()
print(f"[no-grad] loss_orig={loss_orig.item():.6f} loss_cached={loss_c.item():.6f} "
      f"absdiff={abs(loss_orig.item()-loss_c.item()):.2e}", flush=True)

# gradient comparison
trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
probe = ["model.state_proj.weight", "model.action_out_proj.weight",
         "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
         "model.vlm_with_expert.lm_expert.layers.1.self_attn.k_proj.weight",
         "model.action_in_proj.weight"]
probe = [n for n in probe if n in trainable]

policy.zero_grad(set_to_none=True)
loss_o, _ = policy.forward(batch, noise=noise, time=time_t)
loss_o.backward()
go = {n: trainable[n].grad.detach().clone() for n in probe}

policy.zero_grad(set_to_none=True)
img_emb, lang_emb, lang_mask, state, actions = cached_inputs(batch)
kv, fpad = cached.frozen_prefill(img_emb, lang_emb, lang_mask)
loss_c2 = cached.loss_from_batch(kv, fpad, state, actions, batch.get("action_is_pad"))
# use same noise/time for fair grad compare
policy.zero_grad(set_to_none=True)
losses_c = cached.train_forward(kv, fpad, state, actions, noise=noise, time=time_t)
losses_c = losses_c[:, :, :2]
if aip is not None:
    losses_c = losses_c * (~aip).unsqueeze(-1)
    lc = losses_c.sum() / ((~aip).sum() * losses_c.shape[-1]).clamp_min(1)
else:
    lc = losses_c.mean()
lc.backward()
print(f"[grad] loss_orig={loss_o.item():.6f} loss_cached={lc.item():.6f}", flush=True)
for n in probe:
    g1, g2 = go[n], trainable[n].grad
    cos = torch.nn.functional.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
    rel = (g1 - g2).norm().item() / (g1.norm().item() + 1e-12)
    print(f"  {n:70s} cos={cos:.6f} rel_l2={rel:.3e}", flush=True)

# ---- speed benchmark ----
def bench(fn, n=12, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

dl32 = DataLoader(ds, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True,
                  pin_memory=True, drop_last=True)
it = iter(dl32)


def get_batch():
    b = next(it)
    for k in b:
        if "images" in k and b[k].dtype == torch.uint8:
            b[k] = b[k].to(torch.float32) / 255.0
    return pre(b)


opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
policy.train()


def orig_step():
    b = get_batch()
    opt.zero_grad(set_to_none=True)
    loss, _ = policy.forward(b)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 10.0)
    opt.step()


def cached_step():
    b = get_batch()
    opt.zero_grad(set_to_none=True)
    img_emb, lang_emb, lang_mask, state, actions = cached_inputs(b)
    kv, fpad = cached.frozen_prefill(img_emb, lang_emb, lang_mask)
    loss = cached.loss_from_batch(kv, fpad, state, actions, b.get("action_is_pad"))
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 10.0)
    opt.step()


# cached_step still recomputes vision via cached_inputs; benchmark the pure model part separately
def cached_model_only():
    b = get_batch()
    with torch.no_grad():
        img_emb, lang_emb, lang_mask, state, actions = cached_inputs(b)
        kv, fpad = cached.frozen_prefill(img_emb, lang_emb, lang_mask)
    opt.zero_grad(set_to_none=True)
    loss = cached.loss_from_batch(kv, fpad, state, actions, b.get("action_is_pad"))
    loss.backward()
    opt.step()


t_orig = bench(orig_step)
print(f"[bench] original full step (incl vision+preproc): {t_orig:.1f} ms  ({32/(t_orig/1000):.1f} smp/s)", flush=True)
t_cached_full = bench(cached_step)
print(f"[bench] cached step (vision recomputed online):   {t_cached_full:.1f} ms  ({32/(t_cached_full/1000):.1f} smp/s)", flush=True)
print(f"[bench] speedup (cached w/ online vision): {t_orig/t_cached_full:.2f}x", flush=True)
print("peak mem GB:", torch.cuda.max_memory_allocated() / 1e9, flush=True)