"""Create a 1D-adapted init checkpoint from smolvla_base WITHOUT a forward pass.

lerobot-train crashes on action_dim=1 (dataset squeezes the singleton action dim to
(B,50), breaking prepare_action). make_policy only constructs + loads weights (no
forward), so we build the 1D-adapted policy here and save a step-0 checkpoint that
the cached fast trainer resumes from. The fast trainer reads action from the cache
as (N,50,1), which is correct.
"""
import os, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
from pathlib import Path
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.common.train_utils import save_checkpoint, update_last_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lerobot/smolvla_base")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=18000)
    args = ap.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.base)
    cfg.pretrained_path = args.base
    cfg.device = "cuda"
    policy = make_policy(cfg, ds_meta=meta)
    print("action_feature:", policy.config.action_feature, flush=True)
    pre, post = make_smolvla_pre_post_processors(policy.config, meta.stats)

    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=1000, num_decay_steps=args.steps, peak_lr=1e-4, decay_lr=2.5e-6
    ).build(opt, num_training_steps=args.steps)

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=args.dataset_root),
        policy=policy.config, steps=args.steps, batch_size=32, seed=1000)
    ck = Path(args.output_dir) / "checkpoints" / "000000"
    save_checkpoint(ck, 0, train_cfg, policy, opt, sched, pre, post, num_processes=1, batch_size=32)
    update_last_checkpoint(ck)
    print("INIT_OK", ck, flush=True)


if __name__ == "__main__":
    main()