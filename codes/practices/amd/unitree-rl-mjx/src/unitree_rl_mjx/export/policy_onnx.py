"""JAX policy to ONNX for the official deploy runtime.

The trained policy is an observation normalizer followed by an MLP; the
deterministic action is the network's location head. The exported graph bakes
the normalizer in as explicit Sub/Div nodes, so the runtime feeds raw
observations in the deploy.yaml order and reads joint-position actions.

Every export is verified against the JAX policy before the folder is kept.
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax
import numpy as np
import yaml
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from unitree_rl_mjx.export.deploy_config import go2_velocity_deploy_cfg
from unitree_rl_mjx.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_cfg

TOLERANCE = 1e-5
_OPSET = 17


def _policy_layers(params) -> list[tuple[np.ndarray, np.ndarray]]:
  """(kernel, bias) per layer, hidden layers first, location head last."""
  net = params[1]["params"]
  hidden = net["MLP_0"]
  layers = [
    (np.asarray(hidden[name]["kernel"]), np.asarray(hidden[name]["bias"]))
    for name in sorted(hidden, key=lambda n: int(n.split("_")[1]))
  ]
  layers.append(
    (np.asarray(net["Dense_0"]["kernel"]), np.asarray(net["Dense_0"]["bias"]))
  )
  return layers


def build_policy_onnx(params):
  """Assemble the ONNX graph: normalize, hidden MLP with ELU, location head."""
  import onnx
  from onnx import TensorProto, helper, numpy_helper

  mean = np.asarray(params[0].mean["state"], np.float32)
  std = np.asarray(params[0].std["state"], np.float32)
  layers = _policy_layers(params)
  obs_size, act_size = layers[0][0].shape[0], layers[-1][1].shape[0]

  initializers = [
    numpy_helper.from_array(mean, "obs_mean"),
    numpy_helper.from_array(std, "obs_std"),
  ]
  nodes = [
    helper.make_node("Sub", ["obs", "obs_mean"], ["centered"]),
    helper.make_node("Div", ["centered", "obs_std"], ["norm_0"]),
  ]
  x = "norm_0"
  for i, (kernel, bias) in enumerate(layers):
    initializers.append(numpy_helper.from_array(kernel.astype(np.float32), f"w_{i}"))
    initializers.append(numpy_helper.from_array(bias.astype(np.float32), f"b_{i}"))
    last = i == len(layers) - 1
    out = "action" if last else f"dense_{i}"
    nodes.append(helper.make_node("MatMul", [x, f"w_{i}"], [f"matmul_{i}"]))
    nodes.append(helper.make_node("Add", [f"matmul_{i}", f"b_{i}"], [out]))
    if not last:
      nodes.append(helper.make_node("Elu", [out], [f"elu_{i}"], alpha=1.0))
      x = f"elu_{i}"

  # The deploy runtime allocates its input tensor straight from the declared
  # shape, so the batch dimension must be a fixed 1, not symbolic.
  graph = helper.make_graph(
    nodes,
    "go2_velocity_policy",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_size])],
    [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, act_size])],
    initializers,
  )
  model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", _OPSET)], ir_version=10
  )
  onnx.checker.check_model(model)
  return model


def jax_policy_fn(params):
  """The deterministic JAX policy the export must match, from params alone."""
  factory_kwargs = unitree_go2_ppo_cfg()["network_factory"]
  obs_size = {key: value.shape for key, value in params[0].mean.items()}
  act_size = _policy_layers(params)[-1][1].shape[0]
  network = ppo_networks.make_ppo_networks(
    obs_size,
    act_size,
    preprocess_observations_fn=running_statistics.normalize,
    **factory_kwargs,
  )
  inference = ppo_networks.make_inference_fn(network)(params, deterministic=True)
  key = jax.random.PRNGKey(0)  # Unused by the deterministic policy.

  @functools.partial(jax.jit)
  def act(state_obs):
    obs = {
      "state": state_obs,
      "privileged_state": np.zeros(params[0].mean["privileged_state"].shape),
    }
    return inference(obs, key)[0]

  return act


def verify_export(params, onnx_path: Path, observations: np.ndarray) -> float:
  """Max elementwise |JAX - ONNX| over the given observation batch."""
  import onnxruntime

  session = onnxruntime.InferenceSession(
    str(onnx_path), providers=["CPUExecutionProvider"]
  )
  # The graph is batch-1, matching the deploy runtime; run one row at a time.
  onnx_actions = np.vstack(
    [
      session.run(None, {"obs": row[None, :].astype(np.float32)})[0]
      for row in observations
    ]
  )
  jax_actions = np.asarray(jax_policy_fn(params)(observations))
  return float(np.max(np.abs(jax_actions - onnx_actions)))


def export_policy(params_path: str | Path, out_dir: str | Path) -> Path:
  """Write the versioned policy folder; keep it only if verification passes."""
  import onnx

  params = brax_model.load_params(params_path)
  out_dir = Path(out_dir)
  (out_dir / "params").mkdir(parents=True, exist_ok=True)
  (out_dir / "exported").mkdir(parents=True, exist_ok=True)

  onnx_path = out_dir / "exported" / "policy.onnx"
  onnx.save(build_policy_onnx(params), str(onnx_path))
  with (out_dir / "params" / "deploy.yaml").open("w") as f:
    yaml.safe_dump(go2_velocity_deploy_cfg(), f, sort_keys=False)

  # Verification observations drawn from the policy's own input distribution
  # (the running statistics), covering +-3 sigma of what it sees deployed.
  rng = np.random.default_rng(0)
  mean = np.asarray(params[0].mean["state"], np.float32)
  std = np.asarray(params[0].std["state"], np.float32)
  observations = (mean + std * rng.standard_normal((256, mean.shape[0]))).astype(
    np.float32
  )
  error = verify_export(params, onnx_path, observations)
  if error > TOLERANCE:
    for path in (onnx_path, out_dir / "params" / "deploy.yaml"):
      path.unlink()
    raise RuntimeError(f"export verification failed: max |JAX - ONNX| = {error:.2e}")
  print(f"exported {out_dir} (max |JAX - ONNX| = {error:.2e})")
  return out_dir
