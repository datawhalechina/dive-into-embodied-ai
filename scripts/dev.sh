#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

# Ensure LFS is initialized
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local >/dev/null 2>&1
  git lfs pull
else
  echo "WARN: git-lfs not installed. Large files may show as pointers."
  echo "      Install: brew install git-lfs (macOS) / apt install git-lfs (Linux)"
fi

# Install dependencies if needed
if [ ! -d node_modules ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  echo "==> Installing dependencies..."
  npm ci
fi

echo "==> Starting dev server on http://localhost:${1:-3000}"
exec npx docusaurus start --port "${1:-3000}"
