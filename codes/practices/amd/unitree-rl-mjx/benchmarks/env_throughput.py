"""Measure batched env.step throughput for Playground's Go1 joystick task.

Unlike throughput.py (bare physics), this steps the full environment — physics,
observations, and rewards — which is what training actually pays for. Actions
are a fixed random batch; episodes are not reset mid-run, since only step cost
is being measured. Results use the same JSON schema as throughput.py so the
chart tooling renders them unchanged; the manifest records what was measured.

    uv run python benchmarks/env_throughput.py --out results.json
    JAX_PLATFORMS=cpu uv run python benchmarks/env_throughput.py --max-batch 512
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import jax
import mujoco
from mujoco_playground import registry

ENV_NAME = "Go1JoystickFlatTerrain"
# Playground 0.2.0 defaults this env to impl="warp" (CUDA-only, no ROCm). Pin
# the XLA implementation so ROCm and CUDA measure the same code path.
ENV_OVERRIDES = {"impl": "jax"}
BATCH_SIZES = (512, 1024, 2048, 4096, 8192)
STEPS = 100
# ROCm's HIP runtime segfaults when ~100 dispatches of this graph queue up
# unsynced (gfx1201, ROCm 7.2.4). Draining the
# queue every few dispatches avoids it; the same cadence runs on every backend
# so the numbers stay comparable.
SYNC_EVERY = 10


def _vram_mb() -> tuple[float | None, str | None]:
  """Device memory currently used, via whichever vendor tool exists."""
  if shutil.which("nvidia-smi"):
    out = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
      capture_output=True,
      text=True,
      check=False,
    ).stdout
    return max(float(line) for line in out.strip().splitlines()), "nvidia-smi"
  if shutil.which("rocm-smi"):
    out = subprocess.run(
      ["rocm-smi", "--showmeminfo", "vram", "--json"],
      capture_output=True,
      text=True,
      check=False,
    ).stdout
    try:
      cards = json.loads(out)
      used = max(
        float(v) for card in cards.values() for k, v in card.items() if "Used" in k
      )
      return used / 1e6, "rocm-smi"
    except (ValueError, AttributeError):
      return None, "rocm-smi (unparsed)"
  return None, None


def measure(env, num_envs: int, steps: int = STEPS) -> dict:
  """Time compilation and steady-state env stepping at one batch size."""
  rng = jax.random.PRNGKey(0)
  reset_keys = jax.random.split(rng, num_envs)
  state = jax.jit(jax.vmap(env.reset))(reset_keys)
  actions = jax.random.uniform(
    jax.random.PRNGKey(1), (num_envs, env.action_size), minval=-1.0, maxval=1.0
  )
  step = jax.jit(jax.vmap(env.step))

  t0 = time.perf_counter()
  state = jax.block_until_ready(step(state, actions))
  compile_s = time.perf_counter() - t0

  t0 = time.perf_counter()
  for i in range(steps):
    state = step(state, actions)
    if (i + 1) % SYNC_EVERY == 0:
      jax.block_until_ready(state)
  jax.block_until_ready(state)
  elapsed = time.perf_counter() - t0

  vram_mb, vram_tool = _vram_mb()
  return {
    "num_envs": num_envs,
    "steps": steps,
    "compile_s": compile_s,
    "wall_s": elapsed,
    "steps_per_s": num_envs * steps / elapsed,
    "vram_mb": vram_mb,
    "vram_tool": vram_tool,
  }


def _measure_one(num_envs: int, steps: int) -> None:
  """Child mode: measure one batch size, emit a machine-readable line."""
  env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)
  device = jax.devices()[0]
  manifest = {
    "measures": "env_step (physics + observations + rewards), Playground " + ENV_NAME,
    "runner": "one process per batch size",
    "backend": jax.default_backend(),
    "device": str(device),
    "device_kind": device.device_kind,
    "jax": jax.__version__,
    "mujoco": mujoco.__version__,
    "host": platform.node(),
    "python": platform.python_version(),
    "xla_flags": os.environ.get("XLA_FLAGS"),
  }
  r = measure(env, num_envs, steps)
  print("RESULT " + json.dumps({"manifest": manifest, "result": r}), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--max-batch", type=int, default=max(BATCH_SIZES))
  parser.add_argument("--steps", type=int, default=STEPS)
  parser.add_argument("--out", type=str, default=None)
  parser.add_argument("--num-envs", type=int, default=None, help="child mode")
  args = parser.parse_args()

  if args.num_envs is not None:
    _measure_one(args.num_envs, args.steps)
    return

  # Each batch size runs in its own process: sequential re-jits in one process
  # segfault ROCm at larger sizes. Same runner
  # on every backend so the comparison holds.
  manifest = None
  results = []
  for num_envs in BATCH_SIZES:
    if num_envs > args.max_batch:
      break
    proc = subprocess.run(
      [
        sys.executable,
        __file__,
        "--num-envs",
        str(num_envs),
        "--steps",
        str(args.steps),
      ],
      capture_output=True,
      text=True,
      check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
      print(
        f"{num_envs:>6} envs  FAILED (exit {proc.returncode}) — dropped", flush=True
      )
      continue
    payload = json.loads(lines[0][len("RESULT ") :])
    manifest = payload["manifest"]
    r = payload["result"]
    results.append(r)
    vram = f"{r['vram_mb']:,.0f} MB" if r["vram_mb"] else "n/a"
    print(
      f"{num_envs:>6} envs  compile {r['compile_s']:6.1f}s  "
      f"{r['steps_per_s']:>12,.0f} env steps/s  vram {vram}",
      flush=True,
    )

  payload = {"manifest": manifest, "results": results}
  if args.out:
    with open(args.out, "w") as f:
      json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
