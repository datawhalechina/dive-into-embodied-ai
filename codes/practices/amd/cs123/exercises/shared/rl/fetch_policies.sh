#!/usr/bin/env bash
# 拉取上游预训 policy 到 shared/rl/policies/，供各 Lab 对比使用。
#
# 来源优先级：
#   1. 本仓库 tmp/lab_source/lab_5_fall_2025/ 已有副本 → 直接 cp
#   2. 上游公开 repo cs123-stanford/lab_5_fall_2025 → raw.githubusercontent.com 下载
#
# 幂等：文件已存在且 size > 0 时跳过。
# 重新拉取：FORCE=1 ./fetch_policies.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$SCRIPT_DIR/policies"
LOCAL_SRC="$SCRIPT_DIR/../../tmp/lab_source/lab_5_fall_2025"
UPSTREAM_BASE="https://raw.githubusercontent.com/cs123-stanford/lab_5_fall_2025/main"

FILES=(test_policy.json)

mkdir -p "$DEST_DIR"

fetch_one() {
    local name="$1"
    local dest="$DEST_DIR/$name"

    if [[ "${FORCE:-0}" != "1" && -s "$dest" ]]; then
        printf "  [skip] %-25s already exists (FORCE=1 to re-fetch)\n" "$name"
        return 0
    fi

    if [[ -s "$LOCAL_SRC/$name" ]]; then
        printf "  [cp  ] %-25s <- %s\n" "$name" "$LOCAL_SRC/$name"
        cp "$LOCAL_SRC/$name" "$dest"
        return 0
    fi

    local url="$UPSTREAM_BASE/$name"
    printf "  [curl] %-25s <- %s\n" "$name" "$url"
    if ! curl -fL --retry 3 -o "$dest" "$url"; then
        echo "    x download failed: $url" >&2
        echo "    manually download $name from https://github.com/cs123-stanford/lab_5_fall_2025 into $DEST_DIR/" >&2
        rm -f "$dest"
        return 1
    fi
}

echo "-> Fetching policies to $DEST_DIR"
for f in "${FILES[@]}"; do
    fetch_one "$f"
done

echo
echo "Done. Current policies/:"
ls -lh "$DEST_DIR"/ 2>/dev/null || true
