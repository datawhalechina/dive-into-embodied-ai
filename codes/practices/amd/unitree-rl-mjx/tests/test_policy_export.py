"""Contract tests for the ONNX policy export.

The exported folder must satisfy the official deploy stack's contract: the
generated deploy.yaml equals the vendored reference contract field for field,
and the ONNX policy matches the deterministic JAX policy to the stated
tolerance on both representative random observations and observations
recorded from the env itself.
"""

import os

import jax
import numpy as np
import pytest
import yaml
from brax.io import model as brax_model

from unitree_rl_mjx.envs import Go2VelocityFlat
from unitree_rl_mjx.export import (
  export_policy,
  go2_velocity_deploy_cfg,
  jax_policy_fn,
  verify_export,
)
from unitree_rl_mjx.export.install import install

PARAMS_PATH = "benchmarks/results/train-go2-r9700-full-seed1/params.bin"
VENDORED_DEPLOY_YAML = (
  "third_party/unitree_rl_mjlab/deploy/robots/go2/config/policy/velocity/v0/"
  "params/deploy.yaml"
)
TOLERANCE = 1e-5

needs_checkpoint = pytest.mark.skipif(
  not os.path.exists(PARAMS_PATH),
  reason="training checkpoint not present; produce it with the training entry",
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
  out_dir = tmp_path_factory.mktemp("policy") / "v0"
  export_policy(PARAMS_PATH, out_dir)
  return out_dir


@pytest.fixture(scope="module")
def params():
  return brax_model.load_params(PARAMS_PATH)


@pytest.mark.skipif(
  not os.path.exists(VENDORED_DEPLOY_YAML),
  reason="pinned unitree_rl_mjlab checkout not present under third_party/",
)
def test_deploy_yaml_matches_vendored_contract():
  # The reference repo ships the deployed contract for this exact task; our
  # generated file must agree with it field for field.
  with open(VENDORED_DEPLOY_YAML) as f:
    vendored = yaml.safe_load(f)
  assert go2_velocity_deploy_cfg() == vendored


@needs_checkpoint
def test_export_writes_the_policy_folder(exported):
  assert (exported / "params" / "deploy.yaml").is_file()
  assert (exported / "exported" / "policy.onnx").is_file()


@needs_checkpoint
def test_onnx_matches_jax_on_representative_observations(exported, params):
  rng = np.random.default_rng(7)
  mean = np.asarray(params[0].mean["state"], np.float32)
  std = np.asarray(params[0].std["state"], np.float32)
  observations = (mean + std * rng.standard_normal((512, mean.shape[0]))).astype(
    np.float32
  )
  error = verify_export(params, exported / "exported" / "policy.onnx", observations)
  assert error < TOLERANCE


@needs_checkpoint
def test_install_places_only_the_versioned_folder(tmp_path):
  checkout = tmp_path / "checkout"
  (checkout / "deploy/robots/go2/config/policy").mkdir(parents=True)
  target = install("go2", PARAMS_PATH, str(checkout), "v1")
  assert (target / "params" / "deploy.yaml").is_file()
  assert (target / "exported" / "policy.onnx").is_file()
  with pytest.raises(FileExistsError):
    install("go2", PARAMS_PATH, str(checkout), "v1")


@needs_checkpoint
def test_onnx_matches_jax_on_recorded_observations(exported, params):
  # Observations from the env under the trained policy itself - the exact
  # distribution the deployed network will see.
  env = Go2VelocityFlat()
  act = jax_policy_fn(params)
  reset, step = jax.jit(env.reset), jax.jit(env.step)
  state = reset(jax.random.PRNGKey(11))
  recorded = [state.obs["state"]]
  for _ in range(30):
    state = step(state, act(state.obs["state"]))
    recorded.append(state.obs["state"])
  observations = np.stack(recorded).astype(np.float32)
  error = verify_export(params, exported / "exported" / "policy.onnx", observations)
  assert error < TOLERANCE
