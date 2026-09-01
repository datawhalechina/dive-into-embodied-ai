"""Regression checks for the experimental ROCm integration."""

import os
from types import SimpleNamespace

import pytest
import warp as wp


@pytest.mark.skipif("+rocm" not in wp.__version__, reason="not the ROCm Warp fork")
def test_rocm_compatibility_hooks_disable_nested_graph_capture():
    from mjlab.sim.sim import Simulation

    import mjlab_microduck  # noqa: F401

    fake_sim = SimpleNamespace(wp_device=SimpleNamespace(is_hip=True))
    assert Simulation._should_use_cuda_graph(fake_sim) is False
    assert os.environ["WP_HIP_GRAPH_ENABLE"] == "0"
    assert hasattr(wp, "context")
