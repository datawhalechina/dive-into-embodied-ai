"""Fast cached SmolVLA trainer (frozen-prefix embedding cache + split-grad forward).

Drop-in continuation of the lerobot-train recipe: same AdamW/cosine/grad-clip,
same EpisodeAwareSampler data order (sample-exact resume), same checkpoint format
(eval_rollout_v5.py and lerobot resume work unchanged). Skips the vision encoder
and image preprocessing by reading cached SigLIP embeddings; backward only flows
through the 51 trainable tokens (1 state + 50 actions).
"""
import os, time, json, math, argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler

from lerobot.datasets import EpisodeAwareSampler, LeRobotDatasetMetadata, compute_sampler_state
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.common.train_utils import save_checkpoint, load_training_state, update_last_checkpoint
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.utils import cycle
from cached_smolvla import CachedSmolVLA

QD = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco"
MERGED = f"{QD}/outputs/dataset_merged51"


class CacheDataset(Dataset):
    def __init__(self, cache_dir):
        self.info = json.load(open(os.path.join(cache_dir, "info.json")))
        self.N = self.info["n_frames"]
        self.img_dim = self.info["img_dim"]
        self.dim = self.info["hidden"]
        # Materialize all meta arrays eagerly: NpzFile lazy zip handles are not fork-safe
        # across DataLoader workers (BadZipFile) and would re-read the zip per item.
        m = np.load(os.path.join(cache_dir, "meta.npz"))
        self.meta = {k: m[k] for k in m.files}
        self.emb_path = os.path.join(cache_dir, "img_emb.u16")
        self.mm = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.mm is None:
            self.mm = np.memmap(self.emb_path, dtype=np.uint16, mode="r",
                                shape=(self.N, self.img_dim, self.dim))
        return {
            "img": torch.from_numpy(np.ascontiguousarray(self.mm[idx])),
            "state": torch.from_numpy(self.meta["state"][idx]),
            "action": torch.from_numpy(self.meta["action"][idx]),
            "pad": torch.from_numpy(self.meta["action_is_pad"][idx]),
            "task": int(self.meta["task_index"][idx]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dataset/mtl_data/quaddle_vla/cache_emb_v51")
    ap.add_argument("--resume", default=f"{QD}/outputs/smolvla_quaddle_v51_18k/checkpoints/last")
    ap.add_argument("--repo-id", default="local/quaddle_nav_352v51")
    ap.add_argument("--dataset-root", default=MERGED)
    ap.add_argument("--output_dir", default="/dataset/mtl_data/quaddle_vla/runs/v51_30k_fast")
    ap.add_argument("--steps", type=int, default=30000, help="total target step (resume continues from ckpt step)")
    ap.add_argument("--decay", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save_freq", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--no_resume_optim", action="store_true", help="load weights only, fresh optimizer/scheduler")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    resume = Path(args.resume)
    pretrained = resume / "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(str(pretrained)).to("cuda")
    cfg = policy.config
    cached = CachedSmolVLA(policy)
    if args.compile:
        cached.frozen_prefill = torch.compile(cached.frozen_prefill, mode="max-autotune-no-cudagraphs")
        cached.train_forward = torch.compile(cached.train_forward, mode="max-autotune-no-cudagraphs")

    ds = CacheDataset(args.cache)
    lang_emb_table = torch.from_numpy(ds.meta["lang_emb"]).view(torch.bfloat16).cuda()
    lang_mask_table = torch.from_numpy(ds.meta["lang_mask"]).bool().cuda()

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    pre, post = make_smolvla_pre_post_processors(cfg, meta.stats)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    # lerobot builds AdamW over ALL parameters (frozen included) via get_optim_params();
    # match that structure so the saved optimizer state loads index-aligned.
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched_cfg = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=1000, num_decay_steps=args.decay, peak_lr=1e-4, decay_lr=2.5e-6)
    scheduler = sched_cfg.build(opt, num_training_steps=args.steps)

    start_step = 0
    if not args.no_resume_optim and (resume / "training_state").is_dir():
        start_step, opt, scheduler = load_training_state(resume, opt, scheduler)
        print(f"resumed optimizer/scheduler/rng at step {start_step}", flush=True)
    # Reposition the NEW decay schedule at start_step. lerobot convention: the step
    # that produces index N uses lr(N-1), so the first new step (start_step+1) uses
    # lr(start_step). Set that lr explicitly; the loop's scheduler.step() advances it.
    scheduler.last_epoch = start_step
    scheduler._step_count = start_step + 1
    base_lr = scheduler.base_lrs[0]
    resume_lr = base_lr * scheduler.lr_lambdas[0](start_step)
    for pg in opt.param_groups:
        pg["lr"] = resume_lr
    print(f"start lr {resume_lr:.3e} (decay={args.decay}, target={args.steps})", flush=True)

    sampler = EpisodeAwareSampler(
        meta.episodes["dataset_from_index"], meta.episodes["dataset_to_index"],
        shuffle=True, seed=args.seed)
    if start_step > 0:
        st = compute_sampler_state(start_step, len(sampler), args.batch, 1)
        sampler.load_state_dict(st)
        print(f"sampler resume epoch {st['epoch']} offset {st['start_index']}", flush=True)
    batch_sampler = BatchSampler(sampler, args.batch, drop_last=True)
    dl = DataLoader(ds, batch_sampler=batch_sampler, num_workers=args.workers,
                    persistent_workers=args.workers > 0, pin_memory=True)

    train_cfg = TrainPipelineConfig.from_pretrained(str(pretrained))
    train_cfg.steps = args.steps
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_f = open(out / "metrics.jsonl", "a")

    policy.train()
    step = start_step
    t_log = time.perf_counter()
    n_log = 0
    loss_acc = 0.0
    print(f"=== cached training start: step {step} -> {args.steps}, batch {args.batch} ===", flush=True)
    for batch in cycle(dl):
        if step >= args.steps:
            break
        img = batch["img"].cuda(non_blocking=True).view(torch.bfloat16)
        task = batch["task"].cuda(non_blocking=True)
        lang_emb = lang_emb_table[task]
        lang_mask = lang_mask_table[task]
        state = pad_vector(batch["state"].cuda(non_blocking=True), cfg.max_state_dim)
        action = pad_vector(batch["action"].cuda(non_blocking=True), cfg.max_action_dim)
        pad = batch["pad"].cuda(non_blocking=True)

        kv, fpad = cached.frozen_prefill(img, lang_emb, lang_mask)
        loss = cached.loss_from_batch(kv, fpad, state, action, pad)

        if not math.isfinite(loss.item()):
            print(f"!!! non-finite loss at step {step + 1}, skipping update (weights protected)", flush=True)
            opt.zero_grad(set_to_none=True)
            step += 1
            scheduler.step()
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        opt.step()
        scheduler.step()
        step += 1
        n_log += 1
        loss_acc += loss.item()

        if step % args.log_every == 0:
            dt = time.perf_counter() - t_log
            sps = n_log * args.batch / dt
            rec = {"step": step, "loss": loss_acc / n_log, "lr": opt.param_groups[0]["lr"],
                   "grad_norm": gn.item(), "smp_s": sps, "ms_step": dt / n_log * 1000,
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            print(f"step {step} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} "
                  f"{rec['ms_step']:.1f} ms/step {sps:.0f} smp/s mem {rec['mem_gb']:.1f}G", flush=True)
            t_log = time.perf_counter(); n_log = 0; loss_acc = 0.0

        if step % args.save_freq == 0 or step == args.steps:
            ck = out / "checkpoints" / f"{step:06d}"
            save_checkpoint(ck, step, train_cfg, policy, opt, scheduler, pre, post,
                            num_processes=1, batch_size=args.batch)
            update_last_checkpoint(ck)
            print(f"saved checkpoint {ck}", flush=True)

    metrics_f.close()
    print(f"=== training complete at step {step} ===", flush=True)


if __name__ == "__main__":
    main()