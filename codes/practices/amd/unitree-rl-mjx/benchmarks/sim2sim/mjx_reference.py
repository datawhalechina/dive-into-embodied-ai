"""MJX-side reference for the sim2sim tracking table.

Rolls the exported policy's source checkpoint in the training environment
under the same three fixed commands as the sim2sim run and reports the same
segment metrics (mean body-frame velocity, lin/ang tracking error, transient
excluded). Differences between this table and the sim2sim one isolate the
simulator gap from the policy's own tracking behaviour.

    python mjx_reference.py <params.bin> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model as brax_model
from mujoco.mjx._src import math

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval_policy import _inference_fn

from unitree_rl_mjx.envs import Go2VelocityFlat

COMMANDS = {
  "forward": (1.0, 0.0, 0.0),
  "lateral": (0.0, 0.4, 0.0),
  "turn": (0.0, 0.0, 0.8),
}
STEPS = 400  # 8 s at the 0.02 s control step.
TRANSIENT_STEPS = 100  # Same 2 s transient exclusion as the sim2sim metrics.


def rollout(env, inference_fn, command: tuple[float, float, float]) -> dict:
  reset, step = jax.jit(env.reset), jax.jit(env.step)
  state = reset(jax.random.PRNGKey(3))
  key = jax.random.PRNGKey(0)

  # The training env recomputes command[2] every step from its heading servo
  # (wz = clip(stiffness * heading_error)), so a raw yaw-rate command is
  # injected by holding the target far enough ahead of the current heading
  # that the servo output saturates at the wanted rate. Deploy feeds raw wz
  # with no servo, so this is the closest in-env equivalent.
  stiffness = env._cfg.commands["twist"].heading_control_stiffness

  velocities = []
  for _ in range(STEPS):
    fwd = math.rotate(jnp.array([1.0, 0.0, 0.0]), state.data.qpos[3:7])
    heading_w = jnp.arctan2(fwd[1], fwd[0])
    state.info["command"] = jnp.array(command)
    state.info["heading_target"] = heading_w + command[2] / stiffness * 1.1
    state.info["is_standing"] = jnp.zeros((), bool)
    state.info["steps_until_resample"] = jnp.array(10**9, jnp.int32)
    action, _ = inference_fn(state.obs, key)
    state = step(state, action)
    v_body = math.rotate(
      state.data.qvel[:3], math.quat_inv(state.data.qpos[3:7])
    )
    wz = state.data.qvel[5]
    velocities.append([float(v_body[0]), float(v_body[1]), float(wz)])
    if state.done:
      break

  v = np.array(velocities[TRANSIENT_STEPS:])
  cmd = np.array(command)
  lin_err = np.linalg.norm(cmd[:2] - v[:, :2], axis=1)
  return {
    "command": list(command),
    "steps": len(v),
    "mean_body_velocity": [float(x) for x in v.mean(axis=0)],
    "lin_vel_error_mean": float(lin_err.mean()),
    "ang_vel_error_mean": float(np.abs(cmd[2] - v[:, 2]).mean()),
  }


def main() -> None:
  params_path, out_path = sys.argv[1], sys.argv[2]
  env = Go2VelocityFlat()
  inference_fn = jax.jit(_inference_fn(env, brax_model.load_params(params_path)))
  results = {}
  for name, command in COMMANDS.items():
    results[name] = rollout(env, inference_fn, command)
    r = results[name]
    print(
      f"{name:8s} cmd={command}  actual=({r['mean_body_velocity'][0]:6.3f},"
      f"{r['mean_body_velocity'][1]:6.3f},{r['mean_body_velocity'][2]:6.3f})"
      f"  lin_err={r['lin_vel_error_mean']:.4f}"
      f"  ang_err={r['ang_vel_error_mean']:.4f}"
    )
  Path(out_path).write_text(
    json.dumps({"params": params_path, "results": results}, indent=2) + "\n"
  )
  print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
