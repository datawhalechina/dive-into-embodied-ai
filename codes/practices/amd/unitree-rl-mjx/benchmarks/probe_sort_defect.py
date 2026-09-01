"""Probe device sort correctness across ops, dtypes, and shapes.

gfx1201 intermittently corrupts large device sorts: the
output of `jnp.sort` on a float32 vector can
come back neither ordered nor the same multiset, and differs between runs on
identical input — a race. This suite measures that defect precisely: which ops,
which dtypes, which shapes, and how often, so any jax/ROCm build can be judged
in minutes and every workaround has a removal condition.

    uv run python benchmarks/probe_sort_defect.py --out results.json   # all
    uv run python benchmarks/probe_sort_defect.py --case sort-f32-16384

Failures are probabilistic in size, so every report carries its trial count —
a pass at few trials is weak evidence. Each case runs in its own subprocess:
one compiled shape per process (sequential re-jits crash this card), and a
crashed case must not take the suite down with it.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

TRIALS = 16
CURVE_SIZES = (2048, 3072, 4096, 4608, 5120, 6144, 8192, 12288, 16384)

# The op x dtype x shape matrix, at sizes where the defect is likely (16384)
# and at the training-relevant size (8192). name -> (kind, dtype, n, batched)
CASES = {
  "sort-f32-8192": ("sort", "float32", 8192, False),
  "sort-f32-16384": ("sort", "float32", 16384, False),
  "sort-i32-16384": ("sort", "int32", 16384, False),
  "sort-f32-batched-4x16384": ("sort", "float32", 16384, True),
  "argsort-f32-16384": ("argsort", "float32", 16384, False),
  "argsort-i32-16384": ("argsort", "int32", 16384, False),
  "perm-index-16384": ("perm_index", "int32", 16384, False),
  "perm-array-f32-16384": ("perm_array", "float32", 16384, False),
}
CASES.update({f"curve-{n}": ("sort", "float32", n, False) for n in CURVE_SIZES})


def _run_case(kind, dtype, n, batched, trials):
  """Return per-trial pass/fail for one compiled shape."""
  import jax
  import jax.numpy as jnp
  import numpy as np

  def make_input(seed):
    key = jax.random.PRNGKey(seed)
    shape = (4, n) if batched else (n,)
    if dtype == "int32":
      return jax.random.randint(key, shape, 0, 1 << 30)
    return jax.random.uniform(key, shape)

  if kind == "sort":
    fn = jax.jit(lambda x: jnp.sort(x, axis=-1))

    def check(x, y):
      return np.allclose(np.sort(np.asarray(x), axis=-1), np.asarray(y))

  elif kind == "argsort":
    fn = jax.jit(lambda x: jnp.argsort(x, axis=-1))

    def check(x, y):
      idx = np.asarray(y)
      if not (np.sort(idx, axis=-1) == np.arange(n)).all():
        return False  # Not a permutation of the indices at all.
      gathered = np.take_along_axis(np.asarray(x), idx, axis=-1)
      return bool((np.diff(gathered, axis=-1) >= 0).all())

  elif kind == "perm_index":
    fn = jax.jit(lambda key: jax.random.permutation(key, n))

    def check(x, y):
      del x  # Unused.
      return bool((np.sort(np.asarray(y)) == np.arange(n)).all())

  elif kind == "perm_array":
    # Rows must move as units: column 1 stays twice column 0.
    fn = jax.jit(lambda key, x: jax.random.permutation(key, x))

    def check(x, y):
      del x  # Unused.
      out = np.asarray(y)
      rows_kept = (np.sort(out[:, 0]) == np.arange(n)).all()
      return bool(rows_kept and (out[:, 1] == 2 * out[:, 0]).all())

  failures, failed_seeds = 0, []
  for seed in range(trials):
    if kind == "perm_index":
      y = fn(jax.random.PRNGKey(seed))
      x = None
    elif kind == "perm_array":
      base = jnp.arange(n, dtype=jnp.float32)
      x = jnp.stack([base, 2 * base], axis=1)
      y = fn(jax.random.PRNGKey(seed), x)
    else:
      x = make_input(seed)
      y = fn(x)
    if not check(x, y):
      failures += 1
      failed_seeds.append(seed)
  return failures, failed_seeds


def _child(name, trials):
  import jax

  kind, dtype, n, batched = CASES[name]
  t0 = time.perf_counter()
  failures, failed_seeds = _run_case(kind, dtype, n, batched, trials)
  print(
    "RESULT "
    + json.dumps(
      {
        "case": name,
        "kind": kind,
        "dtype": dtype,
        "n": n,
        "batched": batched,
        "trials": trials,
        "failures": failures,
        "failed_seeds": failed_seeds,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "backend": jax.default_backend(),
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
      }
    )
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", choices=sorted(CASES), default=None)
  parser.add_argument("--trials", type=int, default=TRIALS)
  parser.add_argument("--out", type=str, default=None)
  args = parser.parse_args()

  if args.case:
    _child(args.case, args.trials)
    return

  results = []
  for name in CASES:
    cmd = [sys.executable, __file__, "--case", name, "--trials", str(args.trials)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
    if line is None:
      print(f"{name:28s} CRASHED (exit {proc.returncode})")
      results.append({"case": name, "crashed": True, "exit": proc.returncode})
      continue
    r = json.loads(line[len("RESULT ") :])
    results.append(r)
    verdict = "clean" if r["failures"] == 0 else "CORRUPT"
    print(
      f"{name:28s} {verdict:8s} {r['failures']:>2}/{r['trials']} failed  "
      f"({r['elapsed_s']:.1f}s)"
    )

  if args.out:
    devices = [r for r in results if "device" in r]
    manifest = {
      "suite": "probe_sort_defect",
      "trials": args.trials,
      "xla_flags": os.environ.get("XLA_FLAGS"),
      "backend": devices[0]["backend"] if devices else None,
      "device": devices[0]["device"] if devices else None,
      "jax": devices[0]["jax"] if devices else None,
      "host": platform.node(),
    }
    with open(args.out, "w") as f:
      json.dump({"manifest": manifest, "results": results}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
