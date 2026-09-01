"""Import-level smoke tests for the training and benchmark entries.

These need MuJoCo Playground; they skip where it is absent.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"


def _load(name: str):
  spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_train_entry_builds_published_config():
  entry = _load("train_playground_go1")
  train_fn = entry._train_fn(seed=0, num_timesteps=1000, progress_fn=lambda *a: None)
  assert callable(train_fn)
  assert train_fn.keywords["num_timesteps"] == 1000
  assert train_fn.keywords["num_envs"] == 8192  # Playground's published value.


def test_env_benchmark_batch_ladder():
  bench = _load("env_throughput")
  assert bench.BATCH_SIZES == (512, 1024, 2048, 4096, 8192)
  assert bench.ENV_NAME == "Go1JoystickFlatTerrain"


def test_entries_pin_the_xla_impl():
  # Playground defaults to mujoco-warp, which has no ROCm backend; both
  # entries must select MJX's XLA implementation explicitly.
  for name in ("train_playground_go1", "env_throughput"):
    assert _load(name).ENV_OVERRIDES == {"impl": "jax"}
