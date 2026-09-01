#!/usr/bin/env bash
# Record what was actually installed, so any number produced on this machine can be
# traced back to a toolchain. Sourced by the setup scripts.
set -euo pipefail

write_manifest() {
  local backend="$1"
  local out="env-manifest.txt"

  {
    echo "backend: $backend"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(uname -srm)"
    echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo
    uv run python - <<'PY'
import importlib.metadata as md

for name in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "numpy"):
  try:
    print(f"{name}: {md.version(name)}")
  except md.PackageNotFoundError:
    print(f"{name}: not installed")

import jax

print(f"jax devices: {jax.devices()}")
PY
    echo
    if command -v rocm-smi >/dev/null 2>&1; then
      echo "rocm version: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
      rocm-smi --showproductname 2>/dev/null | sed 's/^/  /' || true
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
        | sed 's/^/gpu: /' || true
    fi
  } > "$out"

  echo "==> wrote $out"
  cat "$out"
}
