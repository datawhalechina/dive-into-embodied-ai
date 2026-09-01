"""Train MuJoCo Playground's Go1 joystick task, unmodified, with brax PPO.

The task, its rewards and its published PPO hyperparameters are used exactly as
Playground ships them, so a failure here indicts the JAX/GPU substrate rather
than this repository's code. The run writes:

    metrics.jsonl      one row per evaluation: step count + scalar metrics
    params.bin         trained brax parameters
    trajectory.npz     qpos sequence of one evaluation episode (render offline)
    run.json           backend, device, versions, wall-clock, seed

    uv run python benchmarks/train_playground_go1.py --seed 0 --out-dir runs/r0

--num-timesteps caps the step budget for plumbing smoke runs only; leave unset
for real training.
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
# dependency, never around this file's own logic, and names that defect;
# evidence and measurements live under benchmarks/results/ (the sort-defect
# summaries and probe runs).
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

# XLA's own sort emitter races on gfx1201 for 1-D sorts in roughly the
# 4.6k-16k band: output arrives with entries duplicated and dropped, differing
# run to run on identical input. brax shuffles its rollout data through
# jax.random.permutation (an argsort of random keys), so in that band the
# shuffle silently corrupts the training data and the gradient goes to NaN.
# At >=32768 elements XLA rewrites the sort to hipCUB, which is correct — so
# permutations in the unsafe band pad their keys up to that size with max-value
# sentinels and argsort there. Stable argsort keeps real keys ahead of the
# sentinels, so the first n slots are exactly a uniform permutation of 0..n-1,
# the same algorithm jax uses, on the healthy kernel. Evidence and removal
# condition: benchmarks/probe_sort_defect.py and its committed results.
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

from brax.io import model as brax_model
from brax.training import pmap as brax_pmap
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

_assert_is_replicated = brax_pmap.assert_is_replicated


def _assert_replicated_unless_trivial(x, debug=None):
  # Replication across one device is vacuously true, and the pmap'd equality
  # check inside miscomputes on ROCm. Multi-device runs keep the real check.
  if jax.local_device_count() > 1:
    _assert_is_replicated(x, debug)


brax_pmap.assert_is_replicated = _assert_replicated_unless_trivial

ENV_NAME = "Go1JoystickFlatTerrain"
# Playground 0.2.0 defaults this env to impl="warp" (CUDA-only, no ROCm). Pin
# the XLA implementation, which is the whole point of running on MJX: one code
# path for AMD and NVIDIA. Task physics and rewards are untouched.
ENV_OVERRIDES = {"impl": "jax"}


def _load_env():
  return registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)


def _train_fn(
  seed: int,
  num_timesteps: int | None,
  progress_fn,
  num_envs: int | None = None,
  overrides: dict[str, int] | None = None,
):
  ppo_params = dict(locomotion_params.brax_ppo_config(ENV_NAME))
  network_factory = functools.partial(
    ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
  )
  if num_timesteps is not None:
    ppo_params["num_timesteps"] = num_timesteps
  if num_envs is not None:
    # Substrate limit escape hatch (e.g. the R9700's HSA runtime cannot host
    # the 8192-env training graph); recorded in run.json when used.
    ppo_params["num_envs"] = num_envs
  # Reshapes the training graph (unroll_length, num_minibatches, batch_size) to
  # tell a substrate defect apart from a task one. Any run that sets these is no
  # longer the published config; run.json records what was changed.
  ppo_params.update(overrides or {})
  return functools.partial(
    ppo_train.train,
    **ppo_params,
    network_factory=network_factory,
    seed=seed,
    progress_fn=progress_fn,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    randomization_fn=registry.get_domain_randomizer(ENV_NAME),
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

  env = _load_env()
  t0 = time.perf_counter()
  make_inference_fn, params, _ = _train_fn(
    args.seed, args.num_timesteps, progress, args.num_envs, overrides
  )(environment=env, eval_env=_load_env())
  wall_s = time.perf_counter() - t0

  brax_model.save_params(out_dir / "params.bin", params)

  episode_length = locomotion_params.brax_ppo_config(ENV_NAME)["episode_length"]
  qpos = _eval_trajectory(
    _load_env(), make_inference_fn, params, args.seed, episode_length
  )
  np.savez(out_dir / "trajectory.npz", qpos=np.stack(qpos))

  device = jax.devices()[0]
  (out_dir / "run.json").write_text(
    json.dumps(
      {
        "env": ENV_NAME,
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
