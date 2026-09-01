"""Train the Go2 flat velocity task with brax PPO.

The run writes:

    metrics.jsonl      one row per evaluation: step count + scalar metrics
    params.bin         trained brax parameters
    trajectory.npz     qpos sequence of one evaluation episode (render offline)
    run.json           backend, device, versions, wall-clock, seed

    python -m unitree_rl_mjx.train.go2_velocity --seed 0 --out-dir runs/go2-0

--num-timesteps caps the step budget for plumbing smoke runs only; leave unset
for full training.
"""

import argparse
import functools
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

##
# Substrate workarounds — this block, plus the assert_is_replicated patch just
# after the brax imports (it needs them). Each routes around a defect in a
# dependency, never around this file's own logic, and names that defect and
# its removal condition.
##

if not hasattr(jax, "device_put_replicated"):
  # Removed in jax 0.11.0; brax 0.14.2 (latest) still calls it in ppo.train.
  # Same semantics: replicate a pytree across devices with a leading device axis.
  def _device_put_replicated(x, devices):
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    stack = lambda leaf: jnp.broadcast_to(leaf, (len(devices), *jnp.shape(leaf)))
    return jax.device_put(jax.tree.map(stack, x), sharding)

  jax.device_put_replicated = _device_put_replicated

# XLA's own sort emitter races on gfx1201 (RDNA4) for 1-D sorts in roughly the
# 4.6k-16k element band: output arrives with entries duplicated and dropped,
# differing run to run on identical input. brax shuffles its rollout data
# through jax.random.permutation (an argsort of random keys), so in that band
# the shuffle silently corrupts the training data and the gradient goes to
# NaN. At >=32768 elements XLA rewrites the sort to hipCUB, which is correct —
# so permutations in the unsafe band pad their keys up to that size with
# max-value sentinels and argsort there. Stable argsort keeps real keys ahead
# of the sentinels, so the first n slots are exactly a uniform permutation of
# 0..n-1, the same algorithm jax uses, on the healthy kernel. Remove once
# 1-D jnp.argsort in this size band returns duplicate-free output on the
# deployed gfx1201 stack.
_SORT_SAFE_BELOW = 4096  # No corruption observed at or below this (16 trials).
_CUB_MIN = 32768  # At/above this, sorts take the hipCUB path, measured clean.
_orig_permutation = jax.random.permutation


def _padded_permutation_index(key, n: int):
  """A uniform permutation of 0..n-1 via an argsort routed to hipCUB."""
  keys = jax.random.bits(key, (n,))
  sentinels = jnp.full((_CUB_MIN - n,), jnp.uint32(0xFFFFFFFF))
  return jnp.argsort(jnp.concatenate([keys, sentinels]))[:n]


def _safe_permutation(key, x, axis=0, independent=False):
  n = x if isinstance(x, int) else jnp.shape(x)[axis]
  if n <= _SORT_SAFE_BELOW or n >= _CUB_MIN or independent:
    return _orig_permutation(key, x, axis=axis, independent=independent)
  idx = _padded_permutation_index(key, n)
  return idx if isinstance(x, int) else jnp.take(x, idx, axis=axis)


jax.random.permutation = _safe_permutation

from brax.envs.base import Wrapper as _BraxWrapper
from brax.io import model as brax_model
from brax.training import pmap as brax_pmap
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from mujoco_playground import wrapper

from unitree_rl_mjx.envs import Go2VelocityFlat
from unitree_rl_mjx.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_cfg

_assert_is_replicated = brax_pmap.assert_is_replicated


def _assert_replicated_unless_trivial(x, debug=None):
  # Replication across one device is vacuously true, and the pmap'd equality
  # check inside miscomputes on gfx1201's stack. Multi-device runs keep the
  # real check.
  if jax.local_device_count() > 1:
    _assert_is_replicated(x, debug)


brax_pmap.assert_is_replicated = _assert_replicated_unless_trivial


class _TimeoutFromTruncation(_BraxWrapper):
  """Expose the episode wrapper's truncation flag under brax PPO's key.

  With bootstrap_on_timeout, brax PPO reads `info["time_out"]` to bootstrap
  the value at truncated (not failed) episode ends — the same bootstrapping
  rsl_rl applies to timeouts. The episode wrapper already computes exactly
  this as `info["truncation"]`.
  """

  def reset(self, rng):
    state = self.env.reset(rng)
    state.info["time_out"] = jnp.zeros_like(state.info["truncation"])
    return state

  def step(self, state, action):
    state = self.env.step(state, action)
    state.info["time_out"] = state.info["truncation"]
    return state


def _wrap_env(env, **kwargs):
  return _TimeoutFromTruncation(wrapper.wrap_for_brax_training(env, **kwargs))


def _train_fn(
  env: Go2VelocityFlat,
  seed: int,
  num_timesteps: int | None,
  progress_fn,
  num_envs: int | None = None,
  overrides: dict[str, int] | None = None,
):
  ppo_params = unitree_go2_ppo_cfg()
  network_factory = functools.partial(
    ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
  )
  if num_timesteps is not None:
    ppo_params["num_timesteps"] = num_timesteps
  if num_envs is not None:
    # Hardware-capacity escape hatch (some devices cannot host the 4096-env
    # graph); recorded in run.json when used.
    ppo_params["num_envs"] = num_envs
  ppo_params.update(overrides or {})
  return functools.partial(
    ppo_train.train,
    **ppo_params,
    network_factory=network_factory,
    seed=seed,
    progress_fn=progress_fn,
    wrap_env_fn=_wrap_env,
    randomization_fn=env.domain_randomize_fn(),
  )


def _eval_trajectory(env, make_inference_fn, params, seed: int, num_steps: int):
  """Roll one episode with the trained policy, returning the qpos sequence."""
  inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
  reset = jax.jit(env.reset)
  step = jax.jit(env.step)
  rng = jax.random.PRNGKey(seed)
  state = reset(rng)
  qpos = [state.data.qpos]
  for _ in range(num_steps):
    rng, key = jax.random.split(rng)
    action, _ = inference_fn(state.obs, key)
    state = step(state, action)
    qpos.append(state.data.qpos)
    if state.done:
      break
  return qpos


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", type=str, required=True)
  parser.add_argument("--num-timesteps", type=int, default=None)
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument(
    "--ppo-override",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Integer PPO hyperparameter override, repeatable (diagnostics only).",
  )
  args = parser.parse_args()
  overrides = {k: int(v) for k, v in (o.split("=", 1) for o in args.ppo_override)}

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  metrics_path = out_dir / "metrics.jsonl"
  metrics_path.write_text("")

  def progress(num_steps: int, metrics: dict) -> None:
    row = {"step": num_steps}
    for k, v in metrics.items():
      try:
        row[k] = float(v)
      except (TypeError, ValueError):
        pass  # Non-scalar metric; the curve only needs scalars.
    with metrics_path.open("a") as f:
      f.write(json.dumps(row) + "\n")
    reward = row.get("eval/episode_reward")
    print(f"{num_steps:>12,} steps  reward {reward}", flush=True)

  env = Go2VelocityFlat()
  t0 = time.perf_counter()
  make_inference_fn, params, _ = _train_fn(
    env, args.seed, args.num_timesteps, progress, args.num_envs, overrides
  )(environment=env, eval_env=Go2VelocityFlat())
  wall_s = time.perf_counter() - t0

  brax_model.save_params(out_dir / "params.bin", params)

  episode_length = unitree_go2_ppo_cfg()["episode_length"]
  qpos = _eval_trajectory(
    Go2VelocityFlat(), make_inference_fn, params, args.seed, episode_length
  )
  np.savez(out_dir / "trajectory.npz", qpos=np.stack(qpos))

  device = jax.devices()[0]
  (out_dir / "run.json").write_text(
    json.dumps(
      {
        "env": "Go2VelocityFlat",
        "seed": args.seed,
        "num_timesteps_override": args.num_timesteps,
        "num_envs_override": args.num_envs,
        "ppo_overrides": overrides,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "wall_s": wall_s,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "jax": jax.__version__,
        "mujoco": mujoco.__version__,
        "host": platform.node(),
      },
      indent=2,
    )
  )
  print(f"done in {wall_s:,.0f}s -> {out_dir}")


if __name__ == "__main__":
  main()
