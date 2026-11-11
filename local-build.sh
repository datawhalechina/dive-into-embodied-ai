#!/usr/bin/env sh

set -eu

REQUESTED_PORT="${1:-3001}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

case "$REQUESTED_PORT" in
  ''|*[!0-9]*) echo "Port must be an integer from 1 to 65535." >&2; exit 1 ;;
esac
if [ "${#REQUESTED_PORT}" -gt 5 ] || [ "$REQUESTED_PORT" -lt 1 ] || [ "$REQUESTED_PORT" -gt 65535 ]; then
  echo "Port must be an integer from 1 to 65535." >&2
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "Automatic preview restart requires lsof. Install lsof, then retry." >&2
  exit 1
fi

is_port_in_use() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

is_project_preview() {
  preview_cwd=$(lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
  [ "$preview_cwd" = "$SCRIPT_DIR" ] || return 1
  preview_command=$(ps -p "$1" -o args= 2>/dev/null) || return 1
  case "$preview_command" in
    *node_modules/.bin/docusaurus\ serve\ *|*node_modules/@docusaurus/core/bin/docusaurus.mjs\ serve\ *) return 0 ;;
    *) return 1 ;;
  esac
}

stop_existing_preview() {
  listener_pids=$(lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true)
  [ -n "$listener_pids" ] || return 0

  # Check every listener before stopping anything. Never stop another checkout,
  # a dev server, or an unrelated program just because it uses this port.
  for listener_pid in $listener_pids; do
    is_project_preview "$listener_pid" || return 0
  done
  echo "==> Stopping this checkout's preview on port $1..." >&2
  for listener_pid in $listener_pids; do
    kill -TERM "$listener_pid" 2>/dev/null || true
  done

  attempts=0
  while is_port_in_use "$1"; do
    if [ "$attempts" -ge 5 ]; then
      echo "Preview port $1 did not close. Stop the old preview manually, then retry." >&2
      return 1
    fi
    sleep 1
    attempts=$((attempts + 1))
  done
}

find_available_port() {
  port="$1"

  while is_port_in_use "$port"; do
    stop_existing_preview "$port" || return 1
    if ! is_port_in_use "$port"; then
      break
    fi
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

PORT=$(find_available_port "$REQUESTED_PORT")
if [ "$PORT" != "$REQUESTED_PORT" ]; then
  echo "==> Port ${REQUESTED_PORT} belongs to another process; using ${PORT} instead"
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

# Another process could have taken the selected port while the build ran.
if is_port_in_use "$PORT"; then
  echo "Port ${PORT} was occupied during the build. Retry with another port." >&2
  exit 1
fi

echo "==> Serving build output on http://localhost:${PORT}"
exec ./node_modules/.bin/docusaurus serve --dir build --port "${PORT}"
