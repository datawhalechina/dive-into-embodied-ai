"""Build an actor-only warm-start checkpoint for divewall run-4.

The critic obs grew by 3 (wall_rel, critic-only) so the old critic can't be
restored. This builds a NEW runner on the run-4 env (fresh critic), copies
the ACTOR state (mlp + obs normalizer, dims unchanged) from the source ckpt,
and saves it where --agent.resume can load it.

Usage:
  .venv/bin/python scripts/make_actor_warmstart.py \
      --src logs/rsl_rl/microduck_roulade_dive/<run3>/model_XXXX.pt \
      --out logs/rsl_rl/microduck_roulade_dive/warmstart_divewall_run3_actor \
      --iter XXXX
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

TASK = "Mjlab-RouladeDiveWall-Flat-MicroDuck"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=32)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

    rl_cfg = load_rl_cfg(TASK)
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    new = runner_cls(env, asdict(rl_cfg), device=device)

    src = torch.load(args.src, map_location=device, weights_only=False)
    src_actor = src["actor_state_dict"]

    alg = new.alg
    actor = alg.actor
    tgt = actor.state_dict()
    copied = 0
    for k, v in src_actor.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v
            copied += 1
    actor.load_state_dict(tgt)
    print(f"copied {copied}/{len(src_actor)} actor tensors")

    new.current_learning_iteration = args.iter
    out = f"{args.out}/model_{args.iter}.pt"
    import os
    os.makedirs(args.out, exist_ok=True)
    # torch.save happens inside save(); the logger upload afterwards can crash
    # on an offline Logger (no .writer) — tolerate it, the .pt is already on disk.
    try:
        new.save(out, infos={"warmstart_actor_only": True, "src": args.src})
    except Exception as e:
        print(f"[warn] save() logger step failed (ignored): {e}")
    import pathlib
    assert pathlib.Path(out).exists(), "checkpoint file missing after save"
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
