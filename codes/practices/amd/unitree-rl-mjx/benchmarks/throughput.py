"""Measure batched MJX stepping throughput for the Go2.

Reports, per batch size, how long the first (compiling) call takes and how many
physics steps per second the device sustains afterwards. Results go to stdout as
JSON so they can be plotted or diffed across machines.

    uv run python benchmarks/throughput.py --out results.json
    JAX_PLATFORMS=cpu uv run python benchmarks/throughput.py --max-batch 1024
"""

import argparse
import json
import os
import platform
import time

import jax
import jax.numpy as jnp
from mujoco import mjx

from unitree_rl_mjx.assets.robots.unitree_go2 import go2_constants as go2

BATCH_SIZES = (1, 64, 256, 1024, 4096, 8192)
STEPS = 100


def _batched_data(mj_model, model, num_envs: int):
  data = mjx.make_data(mj_model).replace(qpos=jnp.asarray(mj_model.key_qpos[0]))
  return jax.tree.map(
    lambda leaf: jnp.broadcast_to(leaf, (num_envs, *jnp.shape(leaf))), data
  )


def measure(num_envs: int, steps: int = STEPS, spec: str = "flat") -> dict:
  """Time compilation and steady-state stepping at one batch size.

  "flat" steps the unactuated feet-only model; "training" steps the full
  training model (all collision geoms, position servos) with uniform random
  actions around the default pose, which is what a blown-up contact set or
  broken actuator wiring shows up in first.
  """
  if spec == "training":
    mj_model = go2.get_go2_training_spec().compile()
  else:
    mj_model = go2.get_go2_flat_spec().compile()
  model = mjx.put_model(mj_model)
  batch = _batched_data(mj_model, model, num_envs)

  if spec == "training":
    default_ctrl = jnp.asarray(mj_model.key_ctrl[0])

    @jax.jit
    def step(model, batch, key):
      noise = jax.random.uniform(
        key, (num_envs, mj_model.nu), minval=-0.25, maxval=0.25
      )
      batch = batch.replace(ctrl=default_ctrl + noise)
      return jax.vmap(mjx.step, in_axes=(None, 0))(model, batch)

    keys = jax.random.split(jax.random.PRNGKey(0), steps + 1)
    t0 = time.perf_counter()
    batch = jax.block_until_ready(step(model, batch, keys[0]))
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(steps):
      batch = step(model, batch, keys[i + 1])
    jax.block_until_ready(batch)
    elapsed = time.perf_counter() - t0
  else:
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    t0 = time.perf_counter()
    batch = jax.block_until_ready(step(model, batch))
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(steps):
      batch = step(model, batch)
    jax.block_until_ready(batch)
    elapsed = time.perf_counter() - t0

  return {
    "num_envs": num_envs,
    "steps": steps,
    "compile_s": compile_s,
    "wall_s": elapsed,
    "steps_per_s": num_envs * steps / elapsed,
    "sim_s_per_wall_s": num_envs * steps * mj_model.opt.timestep / elapsed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--max-batch", type=int, default=max(BATCH_SIZES))
  parser.add_argument("--steps", type=int, default=STEPS)
  parser.add_argument("--spec", choices=("flat", "training"), default="flat")
  parser.add_argument("--out", type=str, default=None)
  args = parser.parse_args()

  device = jax.devices()[0]
  manifest = {
    "spec": args.spec,
    "backend": jax.default_backend(),
    "device": str(device),
    "device_kind": device.device_kind,
    "jax": jax.__version__,
    "host": platform.node(),
    "python": platform.python_version(),
    "xla_flags": os.environ.get("XLA_FLAGS"),
  }
  print(json.dumps({"manifest": manifest}), flush=True)

  results = []
  for num_envs in BATCH_SIZES:
    if num_envs > args.max_batch:
      break
    r = measure(num_envs, args.steps, args.spec)
    results.append(r)
    print(
      f"{num_envs:>6} envs  compile {r['compile_s']:6.1f}s  "
      f"{r['steps_per_s']:>12,.0f} steps/s  "
      f"{r['sim_s_per_wall_s']:>7.1f}x realtime",
      flush=True,
    )

  payload = {"manifest": manifest, "results": results}
  if args.out:
    with open(args.out, "w") as f:
      json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
