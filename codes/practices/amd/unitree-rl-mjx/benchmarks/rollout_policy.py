"""Roll out a trained policy under one fixed joystick command, saving qpos.

The trajectory saved during training follows whatever command the task samples,
and the task resamples mid-episode, so that clip wanders. Holding the command
fixed shows the gait for one direction.

    uv run python benchmarks/rollout_policy.py runs/r0/params.bin --out fwd.npz
"""

import argparse
import functools

import jax
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ENV_NAME = "Go1JoystickFlatTerrain"
ENV_OVERRIDES = {"impl": "jax"}


def _inference_fn(env, params):
  """Rebuild the policy network the training entry used, then bind the params."""
  cfg = locomotion_params.brax_ppo_config(ENV_NAME)
  network = functools.partial(ppo_networks.make_ppo_networks, **cfg["network_factory"])(
    env.observation_size,
    env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
  )
  return jax.jit(ppo_networks.make_inference_fn(network)(params, deterministic=True))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("params", help="params.bin from a training run")
  parser.add_argument("--out", required=True)
  parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    default=[1.0, 0.0, 0.0],
    metavar=("VX", "VY", "WZ"),
    help="held for the whole episode: forward, lateral, turn",
  )
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)
  inference_fn = _inference_fn(env, brax_model.load_params(args.params))
  reset, step = jax.jit(env.reset), jax.jit(env.step)

  command = jax.numpy.array(args.command)
  rng = jax.random.PRNGKey(args.seed)
  state = reset(rng)
  qpos = [state.data.qpos]
  for _ in range(args.steps):
    # Overwrite every step: the task resamples its own command on a timer.
    state.info["command"] = command
    rng, key = jax.random.split(rng)
    action, _ = inference_fn(state.obs, key)
    state = step(state, action)
    qpos.append(state.data.qpos)
    if state.done:
      break

  np.savez(args.out, qpos=np.stack(qpos))
  print(f"wrote {args.out} ({len(qpos)} steps)")


if __name__ == "__main__":
  main()
