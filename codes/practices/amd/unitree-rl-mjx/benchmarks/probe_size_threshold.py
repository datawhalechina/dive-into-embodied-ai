"""Reproduce the gfx1201 training defect without brax, MJX or a task.

The defect was found in brax's PPO update, where corruption tracks the rollout
data's leading dimension. That observation cannot be applied to a graph we have
not written yet. This strips the setting to its shape — an MLP, a gradient, an
Adam update, scanned over minibatches — so the failing dimension can be swept
directly, and so a candidate ROCm build can be judged in a minute instead of an
hour.

It also times the update, which is the half of training that the env-step
benchmark does not cover.

    uv run python benchmarks/probe_size_threshold.py            # sweep
    uv run python benchmarks/probe_size_threshold.py --leading 8192   # one size

Each size runs in its own process: compiling several sizes in one process is
itself a way to crash this card.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time

# Shapes follow the Playground Go1 PPO config the defect was found with.
LEADING_SIZES = (1024, 2048, 4096, 5120, 6144, 7168, 8192)
UNROLL = 20
FEATURES = 48
LAYERS = (512, 256, 128)
NUM_MINIBATCHES = 16
ITERS = 4


def _init_params(rng, features, layers):
  import jax

  params = []
  sizes = (features, *layers, 1)
  for a, b in itertools.pairwise(sizes):
    rng, key = jax.random.split(rng)
    w = jax.random.normal(key, (a, b)) * (2.0 / a) ** 0.5
    params.append((w, jax.numpy.zeros((b,))))
  return params


def _measure(
  leading, unroll, features, num_minibatches, iters, use_pmap=False, shuffle=False
):
  """Run `iters` Adam updates over data with this leading dimension.

  brax wraps its training step in pmap even on a single device, so `use_pmap`
  mirrors that: same graph, one extra leading device axis.
  """
  import jax
  import jax.numpy as jnp
  import optax

  rng = jax.random.PRNGKey(0)
  params = _init_params(rng, features, LAYERS)
  data = jax.random.normal(jax.random.PRNGKey(1), (leading, unroll, features))
  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)

  def loss_fn(params, batch):
    x = batch
    for w, b in params[:-1]:
      x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return jnp.mean((x @ w + b) ** 2)

  def minibatch_step(carry, batch):
    params, opt_state = carry
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    return (optax.apply_updates(params, updates), opt_state), ()

  def _epoch(params, opt_state, data):
    # Same as brax: optionally permute the leading axis, then split it into
    # minibatches. The permutation is a gather indexed by the very dimension
    # that predicts the failure, which is why it is worth isolating.
    if shuffle:
      data = jax.random.permutation(jax.random.PRNGKey(2), data)
    batched = jnp.reshape(data, (num_minibatches, -1, unroll, features))
    (params, opt_state), _ = jax.lax.scan(minibatch_step, (params, opt_state), batched)
    return params, opt_state

  if use_pmap:
    epoch = jax.pmap(_epoch, axis_name="i")
    add_axis = lambda x: jnp.broadcast_to(x, (jax.local_device_count(), *x.shape))
    params = jax.tree.map(add_axis, params)
    opt_state = jax.tree.map(add_axis, opt_state)
    data = jax.tree.map(add_axis, data)
  else:
    epoch = jax.jit(_epoch)

  params, opt_state = epoch(params, opt_state, data)
  jax.block_until_ready(params)

  t0 = time.perf_counter()
  for _ in range(iters):
    params, opt_state = epoch(params, opt_state, data)
  jax.block_until_ready(params)
  elapsed = time.perf_counter() - t0

  flat = [jnp.sum(w) for w, _ in params]
  checksum = float(sum(float(v) for v in flat))
  finite = all(bool(jnp.all(jnp.isfinite(w))) for w, _ in params)
  return {
    "leading": leading,
    "unroll": unroll,
    "features": features,
    "num_minibatches": num_minibatches,
    "pmap": use_pmap,
    "shuffle": shuffle,
    "finite": finite,
    "checksum": checksum,
    "s_per_epoch": elapsed / iters,
    "backend": jax.default_backend(),
    "device": jax.devices()[0].device_kind,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--leading", type=int, default=None)
  parser.add_argument("--unroll", type=int, default=UNROLL)
  parser.add_argument("--features", type=int, default=FEATURES)
  parser.add_argument("--num-minibatches", type=int, default=NUM_MINIBATCHES)
  parser.add_argument("--iters", type=int, default=ITERS)
  parser.add_argument("--pmap", action="store_true", help="wrap as brax does")
  parser.add_argument(
    "--shuffle", action="store_true", help="permute the leading axis as brax does"
  )
  args = parser.parse_args()

  if args.leading is not None:
    result = _measure(
      args.leading,
      args.unroll,
      args.features,
      args.num_minibatches,
      args.iters,
      args.pmap,
      args.shuffle,
    )
    print("RESULT " + json.dumps(result))
    return

  for leading in LEADING_SIZES:
    cmd = [
      sys.executable,
      __file__,
      "--leading",
      str(leading),
      "--unroll",
      str(args.unroll),
      "--features",
      str(args.features),
      "--num-minibatches",
      str(args.num_minibatches),
      *(["--pmap"] if args.pmap else []),
      *(["--shuffle"] if args.shuffle else []),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
    if line is None:
      print(f"{leading:>6}  FAILED (exit {proc.returncode})")
      continue
    r = json.loads(line[len("RESULT ") :])
    print(
      f"{leading:>6}  {'finite' if r['finite'] else 'NaN':>6}  "
      f"{r['s_per_epoch'] * 1e3:8.1f} ms/epoch  checksum {r['checksum']:+.4f}"
    )


if __name__ == "__main__":
  main()
