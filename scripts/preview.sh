#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

# Ensure LFS files are present
if command -v git-lfs >/dev/null 2>&1; then
  git lfs pull
fi

echo "==> Building site..."
npm run build

echo "==> Serving on http://localhost:${1:-3001}"
exec npx docusaurus serve --port "${1:-3001}"
