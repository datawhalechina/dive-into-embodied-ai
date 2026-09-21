"""Observation-only distributional value model, adapted from Yirzzzz/pi0.6.

The value target travels in the existing trainer's actions tensor. Only this
model interprets it as a scalar label; it never executes the action expert.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import gemma, model, pi0, pi0_config
from openpi.shared import download
from openpi.training import weight_loaders


def two_hot(values, bins=201):
    scaled = (jnp.clip(values, -1.0, 0.0) + 1.0) * (bins - 1)
    low = jnp.floor(scaled).astype(jnp.int32)
    high = jnp.minimum(low + 1, bins - 1)
    weight = scaled - low
    return (1 - weight[..., None]) * jax.nn.one_hot(low, bins) + weight[..., None] * jax.nn.one_hot(high, bins)


@dataclasses.dataclass(frozen=True)
class ValueConfig(pi0_config.Pi0Config):
    num_value_bins: int = 201

    def create(self, rng):
        return ValueModel(self, nnx.Rngs(rng))


class ValueModel(pi0.Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        width = gemma.get_config(config.paligemma_variant).width
        self.num_value_bins = config.num_value_bins
        self.final_norm = nnx.LayerNorm(width, rngs=rngs)
        self.value_fc1 = nnx.Linear(width, width, rngs=rngs)
        self.value_fc2 = nnx.Linear(width, config.num_value_bins, rngs=rngs)

    def value_logits(self, observation, *, rng=None, train=False):
        if train:
            rng, dropout_rng = jax.random.split(rng)
        observation = model.preprocess_observation(rng, observation, train=train)
        tokens, valid, ar = self.embed_prefix(observation)
        (encoded, _), _ = self.PaliGemma.llm(
            [tokens, None],
            mask=pi0.make_attn_mask(valid, ar),
            positions=jnp.cumsum(valid, axis=1) - 1,
        )
        pooled = jnp.sum(encoded.astype(jnp.float32) * valid[..., None], axis=1)
        pooled /= jnp.maximum(valid.sum(axis=1, keepdims=True), 1)
        features = jax.nn.gelu(self.value_fc1(self.final_norm(pooled)))
        if train:
            # Keep RNG state outside the parameter tree: the upstream trainer
            # casts frozen state to bfloat16 and applies EMA to the entire tree.
            keep = jax.random.bernoulli(dropout_rng, 0.9, features.shape)
            features = jnp.where(keep, features / 0.9, 0)
        return self.value_fc2(features)

    def compute_loss(self, rng, observation, actions, *, train=False):
        targets = two_hot(actions[:, 0, 0], self.num_value_bins)
        logits = self.value_logits(observation, rng=rng, train=train)
        return -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1, keepdims=True)

    def predict_value(self, observation):
        probs = jax.nn.softmax(self.value_logits(observation), axis=-1)
        return jnp.sum(probs * jnp.linspace(-1.0, 0.0, self.num_value_bins), axis=-1)

    def sample_actions(self, *args, **kwargs):
        raise NotImplementedError("A value checkpoint cannot serve robot actions; use the SFT or ACP config.")


@dataclasses.dataclass(frozen=True)
class ValueWeightLoader:
    params_path: str

    def load(self, params):
        loaded = model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Only the added value head may be absent. The upstream trainer still
        # validates the entire backbone's keys, shapes and dtypes.
        return weight_loaders._merge_params(loaded, params, missing_regex=r"(final_norm|value_fc1|value_fc2)/.*")
