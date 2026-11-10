#!/usr/bin/env sh

set -eu

REQUESTED_PORT="${1:-3001}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

is_port_in_use() {
  port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi

  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1 || nc -z ::1 "$port" >/dev/null 2>&1
    return $?
  fi

  return 1
}

find_available_port() {
  port="$1"

  while is_port_in_use "$port"; do
    port=$((port + 1))
    if [ "$port" -gt 65535 ]; then
      echo "No available port found starting from ${REQUESTED_PORT}." >&2
      exit 1
    fi
  done

  printf '%s\n' "$port"
}

cd "$SCRIPT_DIR"

# Reuse a downloaded project runtime when the shell selects another Node version.
NODE_VERSION=$(cat .nvmrc)
if [ "$(node --version 2>/dev/null || true)" != "v${NODE_VERSION}" ]; then
  NODE_PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
  NODE_ARCH=$(uname -m)
  case "$NODE_ARCH" in
    x86_64) NODE_ARCH=x64 ;;
    aarch64) NODE_ARCH=arm64 ;;
  esac
  LOCAL_NODE_BIN="${XDG_DATA_HOME:-$HOME/.local/share}/dive-into-embodied-ai/node-v${NODE_VERSION}-${NODE_PLATFORM}-${NODE_ARCH}/bin"
  if [ -x "$LOCAL_NODE_BIN/node" ] && [ -x "$LOCAL_NODE_BIN/npm" ]; then
    export PATH="$LOCAL_NODE_BIN:$PATH"
  fi
fi

if [ "$(node --version 2>/dev/null || true)" != "v${NODE_VERSION}" ]; then
  echo "Node.js ${NODE_VERSION} is required. Run 'nvm install && nvm use' in this directory, or select that version with your Node version manager." >&2
  exit 1
fi

NPM_VERSION=$(node -p 'require("./package.json").engines.npm')
if [ "$(npm --version 2>/dev/null || true)" != "$NPM_VERSION" ]; then
  echo "npm ${NPM_VERSION} is required. Select the npm bundled with Node.js ${NODE_VERSION}." >&2
  exit 1
fi

# A pull can add dependencies even when node_modules already exists.
if [ ! -f node_modules/.package-lock.json ] ||
   [ package-lock.json -nt node_modules/.package-lock.json ] ||
   [ package.json -nt node_modules/.package-lock.json ] ||
   ! npm ls --depth=0 >/dev/null 2>&1; then
  echo "==> Installing locked dependencies..."
  npm run install:locked
fi

# 本地预览不需要真实职位数据,默认走 fixture,避免 prebuild 因缺 JOBS_API_TOKEN 报错
export JOBS_DATA_MODE="${JOBS_DATA_MODE:-fixture}"

echo "==> Building site... (JOBS_DATA_MODE=${JOBS_DATA_MODE})"
npm run build

PORT=$(find_available_port "$REQUESTED_PORT")

if [ "$PORT" != "$REQUESTED_PORT" ]; then
  echo "==> Port ${REQUESTED_PORT} is in use; using ${PORT} instead"
fi

echo "==> Serving build output on http://localhost:${PORT}"
exec ./node_modules/.bin/docusaurus serve --dir build --port "${PORT}"
