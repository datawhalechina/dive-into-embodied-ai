"""RL configuration for Unitree Go2 velocity task.

The values map the task's rsl_rl PPO configuration onto brax PPO. Where the
two frameworks name the same quantity differently, this table is the record:

  rsl_rl                          brax                        value
  ------------------------------  --------------------------  ----------------
  actor/critic hidden_dims        {policy,value}_hidden_...   (512, 256, 128)
  activation elu                  activation                  linen.elu
  obs_normalization               normalize_observations      True
  GaussianDistribution            distribution_type           "normal"
  init_std 1.0 (scalar)           init_noise_std / noise_...  1.0, "scalar"
  value_loss_coef                 vf_loss_coefficient         1.0
  use_clipped_value_loss          clipping_epsilon_value      0.2 (= policy)
  clip_param                      clipping_epsilon            0.2
  entropy_coef                    entropy_cost                0.01
  num_learning_epochs             num_updates_per_batch       5
  num_mini_batches                num_minibatches             4
  learning_rate (adaptive)        learning_rate + ADAPTIVE_KL 1.0e-3 initial
  desired_kl                      desired_kl                  0.01
  gamma                           discounting                 0.99
  lam                             gae_lambda                  0.95
  max_grad_norm                   max_grad_norm               1.0
  num_steps_per_env               unroll_length               24
  num_envs (runner convention)    num_envs / batch_size       4096 / 1024
  max_iterations 10001            num_timesteps               10001*4096*24
  timeout bootstrap (rsl_rl)      bootstrap_on_timeout        True

brax's adaptive-KL rule is the same update as rsl_rl's (lr /1.5 above twice
the desired KL, x1.5 below half of it, clamped to [1e-5, 1e-2]). brax draws
minibatches of whole-env rollouts (batch_size envs x unroll_length steps)
where rsl_rl shuffles individual transitions; this is brax's sampling
convention and is not configurable. num_evals is a logging cadence with no
rsl_rl counterpart.
"""

from __future__ import annotations

from brax.training.agents.ppo import optimizer as ppo_optimizer
from flax import linen


def unitree_go2_ppo_cfg() -> dict:
  """Create brax PPO parameters for the Go2 velocity task."""
  return {
    "num_timesteps": 10001 * 4096 * 24,
    "num_envs": 4096,
    "unroll_length": 24,
    "num_minibatches": 4,
    "batch_size": 1024,
    "num_updates_per_batch": 5,
    "episode_length": 1000,
    "learning_rate": 1.0e-3,
    "learning_rate_schedule": ppo_optimizer.LRSchedule.ADAPTIVE_KL,
    "desired_kl": 0.01,
    "entropy_cost": 0.01,
    "discounting": 0.99,
    "gae_lambda": 0.95,
    "clipping_epsilon": 0.2,
    "clipping_epsilon_value": 0.2,
    "vf_loss_coefficient": 1.0,
    "max_grad_norm": 1.0,
    "normalize_observations": True,
    "bootstrap_on_timeout": True,
    "reward_scaling": 1.0,
    "num_evals": 20,
    "network_factory": {
      "policy_hidden_layer_sizes": (512, 256, 128),
      "value_hidden_layer_sizes": (512, 256, 128),
      "activation": linen.elu,
      "policy_obs_key": "state",
      "value_obs_key": "privileged_state",
      "distribution_type": "normal",
      "noise_std_type": "scalar",
      "init_noise_std": 1.0,
    },
  }
