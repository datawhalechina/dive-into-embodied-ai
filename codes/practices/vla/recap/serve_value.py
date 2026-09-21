"""Serve a RECAP value checkpoint to the LIBERO rollout process."""

import argparse
import os
import pathlib

# Load one value model without reserving most of the GPU before the policy starts.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from examples.recap.config import ValueTarget, get_configs
from openpi import transforms
from openpi.models import model as model_lib
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import nnx_utils
from openpi.training import checkpoints


class ValuePolicy:
    def __init__(self, checkpoint):
        cfg = next(config for config in get_configs() if config.name == "pi05_recap_value")
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data_cfg.asset_id)
        if norm_stats is None:
            raise ValueError(f"Value checkpoint has no norm stats: {checkpoint / 'assets'}")
        self.transform = transforms.compose(
            [
                *data_cfg.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
                *(step for step in data_cfg.model_transforms.inputs if not isinstance(step, ValueTarget)),
            ]
        )
        model = cfg.model.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        model.eval()
        self.predict = nnx_utils.module_jit(model.predict_value)

    def infer(self, observation):
        inputs = self.transform(dict(observation))
        batch = jax.tree.map(lambda item: jnp.asarray(item)[None, ...], inputs)
        value = np.asarray(self.predict(model_lib.Observation.from_dict(batch))).reshape(-1)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"Expected one finite value prediction, got {value}")
        return {"value": float(value[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        parser.error(f"Value checkpoint has no params directory: {checkpoint}")
    policy = ValuePolicy(checkpoint)
    print(f"Serving RECAP value from {checkpoint} on {args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"kind": "recap_value", "checkpoint": str(checkpoint)},
    ).serve_forever()


if __name__ == "__main__":
    main()
