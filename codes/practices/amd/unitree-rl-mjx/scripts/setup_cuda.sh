#!/usr/bin/env bash
# NVIDIA environment: the CUDA JAX build, used for the cross-vendor comparison.
#
# Pinned to the lockfile's jax version so CPU, CUDA and ROCm all run the same
# jax — an unpinned install would float to the latest release and silently fork
# this backend from the other two.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_write_manifest.sh

JAX_CUDA_VERSION="${JAX_CUDA_VERSION:-0.11.0}"

EXTRAS=(--extra dev)
uv sync "${EXTRAS[@]}"
uv pip install "jax[cuda12]==${JAX_CUDA_VERSION}"

write_manifest cuda
