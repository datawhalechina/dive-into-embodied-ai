#!/usr/bin/env bash
# AMD environment: the ROCm JAX build.
#
# Verified 2026-08-05 on a Radeon AI PRO R9700 (gfx1201, RDNA4), ROCm 7.2.4,
# Ubuntu 24.04, Python 3.12: the plugin version below matches the `jax` version
# in the lockfile, so ROCm needs no separate dependency resolution — the same
# pins serve CPU, CUDA and ROCm.
#
# AMD also publishes ROCm-versioned wheels at repo.radeon.com (for 7.2.4:
# jax_rocm7_plugin 0.8.2+rocm7.2.4). Those trail the PyPI releases and would pull
# an older jax; prefer PyPI unless a specific ROCm build is required.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_write_manifest.sh

JAX_ROCM_VERSION="${JAX_ROCM_VERSION:-0.11.0}"

EXTRAS=(--extra dev)
uv sync "${EXTRAS[@]}"
uv pip install \
  "jax-rocm7-plugin==${JAX_ROCM_VERSION}" \
  "jax-rocm7-pjrt==${JAX_ROCM_VERSION}"

# gfx1201 is the discrete card; the integrated GPU (gfx1036) must not be selected.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

write_manifest rocm
echo "==> expect: backend gpu, devices [RocmDevice(id=0)]"
