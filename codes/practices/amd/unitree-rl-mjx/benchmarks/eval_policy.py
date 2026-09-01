"""Evaluate a trained Go2 velocity policy on held-out commands.

Rolls a batch of full-length episodes with the deterministic policy under
evaluation seeds disjoint from training, and reports convergence metrics:
velocity-tracking errors, episode survival, per-foot air-time regularity,
and mean episode return. The numbers are written as JSON next to the params
file so each run directory carries its own evaluation.

    uv run python benchmarks/eval_policy.py runs/r0/params.bin

The evaluation is deterministic: same params and seeds reproduce the same
metrics.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco.mjx._src import math

from unitree_rl_mjx.envs import Go2VelocityFlat
from unitree_rl_mjx.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_cfg

# Frozen 2026-08-08 from the full-budget CUDA baseline
# (benchmarks/results/train-go2-a40-full-seed0: lin 0.2220, ang 0.0700,
# survival 1.00, air-time CV 0.2663, return 58.82). Rule: errors and CV at
# 1.25x the baseline value; survival allows one lost episode in 32; return
# at 90% of baseline. Direction: "<=" passes at or below, ">=" at or above.
THRESHOLDS: dict[str, tuple[str, float | None]] = {
  "lin_vel_error_mean": ("<=", 0.278),
  "ang_vel_error_mean": ("<=", 0.0875),
  "survival_rate": (">=", 0.95),
  "air_time_cv": ("<=", 0.333),
  "episode_return_mean": (">=", 52.9),
}


def _inference_fn(env, params):
  """Rebuild the policy network the training entry used, then bind the params."""
  factory_kwargs = unitree_go2_ppo_cfg()["network_factory"]
  network = ppo_networks.make_ppo_networks(
    env.observation_size,
    env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    **factory_kwargs,
  )
  return ppo_networks.make_inference_fn(network)(params, deterministic=True)


def _rollout(env, inference_fn, seeds: np.ndarray, num_steps: int):
  """Roll one episode per seed, batched; returns stacked per-step records.

  Records follow the reward's measurement convention: the tracking error at a
  step compares the post-step body-frame velocity against the post-step
  command, and a step counts while the episode is alive entering it (the
  terminating step included).
  """
  vreset = jax.vmap(env.reset)
  vstep = jax.vmap(env.step)
  policy_key = jax.random.PRNGKey(0)  # Unused by the deterministic policy.

  def body(carry, _):
    state, alive = carry
    swing_time = state.info["current_air_time"]
    action, _ = inference_fn(state.obs, policy_key)
    state = vstep(state, action)
    quat = state.data.qpos[:, 3:7]
    lin_vel_b = jax.vmap(lambda v, q: math.rotate(v, math.quat_inv(q)))(
      state.data.qvel[:, 0:3], quat
    )
    command = state.info["command"]
    landed = state.info["first_contact"] & alive[:, None]
    record = {
      "lin_vel_error": jnp.linalg.norm(command[:, :2] - lin_vel_b[:, :2], axis=-1),
      "ang_vel_error": jnp.abs(command[:, 2] - state.data.qvel[:, 5]),
      "reward": state.reward,
      "alive": alive,
      "landed": landed,
      "swing_time": jnp.where(landed, swing_time, 0.0),
    }
    return (state, alive & (state.done == 0)), record

  def run(rngs):
    state = vreset(rngs)
    alive = jnp.ones(rngs.shape[0], dtype=bool)
    (_, alive), records = jax.lax.scan(body, (state, alive), None, length=num_steps)
    return records, alive

  rngs = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
  records, survived = jax.jit(run)(rngs)
  return jax.tree.map(np.asarray, records), np.asarray(survived)


def _metrics(records: dict, survived: np.ndarray) -> dict:
  alive = records["alive"]  # (steps, episodes)
  alive_steps = alive.sum()

  def alive_mean(x):
    return float(np.where(alive, x, 0.0).sum() / alive_steps)

  air_time_cvs = []
  for foot in range(4):
    landed = records["landed"][:, :, foot]
    samples = records["swing_time"][:, :, foot][landed]
    air_time_cvs.append(float(samples.std() / samples.mean()))
  return {
    "lin_vel_error_mean": alive_mean(records["lin_vel_error"]),
    "ang_vel_error_mean": alive_mean(records["ang_vel_error"]),
    "survival_rate": float(survived.mean()),
    "air_time_cv": float(np.mean(air_time_cvs)),
    "air_time_cv_per_foot": air_time_cvs,
    "episode_return_mean": float(
      np.where(alive, records["reward"], 0.0).sum(axis=0).mean()
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("params", help="params.bin from a training run")
  parser.add_argument("--episodes", type=int, default=32)
  parser.add_argument(
    "--seed-base",
    type=int,
    default=10_000,
    help="episode i uses seed seed-base + i; keep disjoint from training seeds",
  )
  parser.add_argument("--out", default=None, help="default: eval.json next to params")
  args = parser.parse_args()

  env = Go2VelocityFlat()
  inference_fn = _inference_fn(env, brax_model.load_params(args.params))
  num_steps = unitree_go2_ppo_cfg()["episode_length"]
  seeds = args.seed_base + np.arange(args.episodes)
  records, survived = _rollout(env, inference_fn, seeds, num_steps)
  metrics = _metrics(records, survived)

  verdicts = {}
  for name, (direction, threshold) in THRESHOLDS.items():
    if threshold is None:
      verdicts[name] = None
      status = "no threshold set"
    else:
      ok = (
        metrics[name] <= threshold if direction == "<=" else metrics[name] >= threshold
      )
      verdicts[name] = bool(ok)
      status = f"{'PASS' if ok else 'FAIL'} ({direction} {threshold})"
    print(f"{name:>24}: {metrics[name]:8.4f}  {status}")

  out = Path(args.out) if args.out else Path(args.params).parent / "eval.json"
  out.write_text(
    json.dumps(
      {
        "params": str(args.params),
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "episode_steps": num_steps,
        "metrics": metrics,
        "thresholds": {k: v for k, (_, v) in THRESHOLDS.items()},
        "passed": verdicts,
      },
      indent=2,
    )
  )
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
