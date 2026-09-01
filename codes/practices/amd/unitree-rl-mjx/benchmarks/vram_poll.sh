#!/usr/bin/env bash
# Poll GPU memory use to CSV until a watched PID exits.
#
#   vram_poll.sh <pid> <out.csv> [interval_s] [nvidia_gpu_index]
#
# Uses whichever vendor tool exists. Rows: unix epoch seconds, used MiB.
set -euo pipefail

pid="$1"
out="$2"
interval="${3:-2}"
gpu_index="${4:-}"

echo "t_unix_s,used_mib" > "$out"
while kill -0 "$pid" 2>/dev/null; do
  if command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi ${gpu_index:+-i "$gpu_index"} \
      --query-gpu=memory.used --format=csv,noheader,nounits \
      | sort -n | tail -1)
  else
    # rocm-smi reports VRAM use in bytes.
    used=$(rocm-smi --showmeminfo vram --json 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(max(int(v) for c in d.values() for k,v in c.items() if "Used" in k)//(1024*1024))')
  fi
  echo "$(date +%s),$used" >> "$out"
  sleep "$interval"
done
