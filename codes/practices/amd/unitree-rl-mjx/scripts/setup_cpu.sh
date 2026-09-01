#!/usr/bin/env bash
# Development and CI environment: CPU-only JAX.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_write_manifest.sh

uv sync --extra dev

write_manifest cpu
echo "==> run the tests with: uv run pytest"
