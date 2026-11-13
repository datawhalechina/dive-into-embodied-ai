"""Build the frozen image-embedding cache for the v5.1 merged dataset.

Stores per-frame SigLIP+connector embeddings (post x sqrt(hidden) scaling, bf16)
as a uint16 memmap, plus normalized state/action chunks, padding masks and
per-task language embeddings. One-time cost (~15 min); training then skips the
vision encoder (266 ms/step) and all image preprocessing entirely.
"""
import os, time, json, math, argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
CKPT = f"{QD}/outputs/smolvla_quaddle_v51_18k/checkpoints/last/pretrained_model"
MERGED = f"{QD}/outputs/dataset_merged51"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/dataset/mtl_data/quaddle_vla/cache_emb_v51")
    ap.add_argument("--repo-id", default="local/quaddle_nav_352v51")
    ap.add_argument("--dataset-root", default=MERGED)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    policy = SmolVLAPolicy.from_pretrained(CKPT).to("cuda").eval()
    cfg = policy.config
    vlm = policy.model.vlm_with_expert
    dim = vlm.config.text_config.hidden_size
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root,
                        delta_timestamps=resolve_delta_timestamps(cfg, meta), return_uint8=True)
    pre, _ = make_smolvla_pre_post_processors(cfg, ds.meta.stats)
    N = len(ds)
    n_cams = len(cfg.image_features)
    tok_per_img = 64
    img_dim = n_cams * tok_per_img
    print(f"frames={N} cams={n_cams} img_tokens={img_dim} hidden={dim}", flush=True)

    emb_path = os.path.join(args.out, "img_emb.u16")
    emb_mm = np.memmap(emb_path, dtype=np.uint16, mode="w+", shape=(N, img_dim, dim))
    state_arr = np.zeros((N, 8), dtype=np.float32)
    action_arr = np.zeros((N, 50, 2), dtype=np.float32)
    pad_arr = np.zeros((N, 50), dtype=bool)
    task_arr = np.zeros((N,), dtype=np.uint8)
    ep_arr = np.zeros((N,), dtype=np.int32)
    lang_tokens_by_task = {}
    lang_mask_by_task = {}

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    persistent_workers=args.workers > 0, pin_memory=True, drop_last=False)
    offset = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            for k in batch:
                if "images" in k and batch[k].dtype == torch.uint8:
                    batch[k] = batch[k].to(torch.float32) / 255.0
            batch = pre(batch)
            images, _ = policy.prepare_images(batch)
            embs = []
            for img in images:
                ie = vlm.embed_image(img)
                ie = ie * torch.tensor(dim ** 0.5, dtype=ie.dtype, device=ie.device)
                embs.append(ie)
            img_emb = torch.cat(embs, dim=1)  # [B,img_dim,dim] bf16
            B = img_emb.shape[0]
            emb_mm[offset:offset + B] = img_emb.view(torch.uint16).cpu().numpy()
            st = batch[OBS_STATE]
            st = st[:, -1, :] if st.ndim > 2 else st
            state_arr[offset:offset + B] = st.cpu().numpy()
            action_arr[offset:offset + B] = batch[ACTION].cpu().numpy()
            pad_arr[offset:offset + B] = batch["action_is_pad"].cpu().numpy()
            ti = batch["task_index"].cpu().numpy()
            task_arr[offset:offset + B] = ti
            ep_arr[offset:offset + B] = batch["episode_index"].cpu().numpy()
            lt = batch[OBS_LANGUAGE_TOKENS].cpu().numpy()
            lm = batch[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy()
            for j in range(B):
                t = int(ti[j])
                if t not in lang_tokens_by_task:
                    lang_tokens_by_task[t] = lt[j]
                    lang_mask_by_task[t] = lm[j]
            offset += B
            if bi % 50 == 0:
                el = time.time() - t0
                print(f"batch {bi} frames {offset}/{N} {el:.0f}s "
                      f"({offset/el:.0f} fr/s, eta {(N-offset)/(offset/el)/60:.1f} min)", flush=True)
    emb_mm.flush()
    del emb_mm

    n_tasks = len(lang_tokens_by_task)
    lang_emb = np.zeros((n_tasks, 48, dim), dtype=np.uint16)
    lang_mask = np.zeros((n_tasks, 48), dtype=bool)
    with torch.no_grad():
        for t in sorted(lang_tokens_by_task):
            toks = torch.from_numpy(lang_tokens_by_task[t]).long().unsqueeze(0).cuda()
            le = vlm.embed_language_tokens(toks) * math.sqrt(dim)
            lang_emb[t] = le[0].view(torch.uint16).cpu().numpy()
            lang_mask[t] = lang_mask_by_task[t]
    np.savez(os.path.join(args.out, "meta.npz"),
             state=state_arr, action=action_arr, action_is_pad=pad_arr,
             task_index=task_arr, episode_index=ep_arr,
             lang_emb=lang_emb, lang_mask=lang_mask)
    info = {"n_frames": N, "img_dim": img_dim, "hidden": dim, "n_cams": n_cams,
            "tok_per_img": tok_per_img, "dtype": "bfloat16 (stored as uint16 bits)",
            "n_tasks": n_tasks, "chunk": 50, "state_dim": 8, "action_dim": 2,
            "dataset": args.repo_id, "ckpt": CKPT}
    json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=2)
    print(f"DONE {offset} frames in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()