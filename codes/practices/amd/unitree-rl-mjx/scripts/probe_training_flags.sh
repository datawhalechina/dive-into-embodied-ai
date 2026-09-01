#!/usr/bin/env bash
# Run the short training repro once per XLA flag setting and classify each run.
#
# Used to narrow down a gfx1201 training defect: with command buffers on the
# first training epoch
# crashes the HSA runtime, with them off it drives the parameters to NaN. Each
# setting runs in its own process (a crashed one must not poison the next), and
# the outcome is decided from the eval rewards, not from the exit code alone:
#
#   PASS    every eval finite and the last one above 0.5 (learning happened)
#   NAN     an eval reward is not finite
#   FLAT    all finite but no learning — inconclusive, not a fix
#   CRASH-N the process died with exit status N
#
#   ./scripts/probe_training_flags.sh runs/probe-2026-08-08          # all of them
#   ./scripts/probe_training_flags.sh runs/shapes unroll-10 batch-128 # named subset
#
# Roughly ten minutes per setting on the R9700; run it under nohup and read
# summary.tsv afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:?usage: probe_training_flags.sh <out-dir> [name ...]}"
shift
WANTED=("$@")
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-3300000}"

#   name | XLA flags | extra arguments to the training entry
#
# Command buffers are off in most probes: with them on the run segfaults before
# it can say anything about the NaN. The baseline entry is the known-bad control
# — if it does not reproduce, the rest of the ladder means nothing. A probe that
# needs the default (capture on) simply leaves the flag out.
#
# A trailing --num-envs in a probe's arguments wins over the script-level one:
# argparse keeps the last occurrence.
#
# Blocks: (1) code generation, (2) training-graph shape, (3) which shape number
# actually matters. brax builds rollout data with leading dimension
# batch_size * num_minibatches and scans it in leading_dim/num_envs steps, so at
# a fixed num_envs those two move together; block 3 separates them.
BUFFERS_OFF="--xla_gpu_enable_command_buffer="
PROBES=(
  "baseline|$BUFFERS_OFF|"
  "fast-min-max-off|$BUFFERS_OFF --xla_gpu_enable_fast_min_max=false|"
  "block-level-fusion-off|$BUFFERS_OFF --xla_gpu_experimental_enable_fusion_block_level_rewriter=false|"
  "heroless-priority-fusion-off|$BUFFERS_OFF --xla_gpu_experimental_enable_triton_heroless_priority_fusion=false|"
  "dynamic-slice-fusion-off|$BUFFERS_OFF --xla_gpu_enable_dynamic_slice_fusion=false|"
  "deterministic-ops|$BUFFERS_OFF --xla_gpu_deterministic_ops=true|"
  "no-excess-precision|$BUFFERS_OFF --xla_allow_excess_precision=false|"
  "scheduler-off|$BUFFERS_OFF --xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_enable_while_loop_double_buffering=false|"
  "unroll-10|$BUFFERS_OFF|--ppo-override unroll_length=10"
  "minibatches-8|$BUFFERS_OFF|--ppo-override num_minibatches=8"
  "batch-128|$BUFFERS_OFF|--ppo-override batch_size=128"
  # leading dim 8192 (as baseline) but a shorter rollout scan: if the leading
  # dimension is what matters this still NaNs, if the scan length is, it passes.
  "envs-2048|$BUFFERS_OFF|--num-envs 2048"
  # leading dim 4096 at the same scan length as envs-2048 — the paired control.
  "minibatches-16|$BUFFERS_OFF|--ppo-override num_minibatches=16"
  # leading dim 8192 with a longer scan, the other side of envs-2048.
  "envs-512|$BUFFERS_OFF|--num-envs 512"
  # does a passing shape also survive command-buffer capture? If so, one size
  # threshold explains both the NaN and the segfault, and the workaround drops.
  # (It does not: a leading dimension of 2048 still segfaults with capture on.)
  "minibatches-8-cmdbuf||--ppo-override num_minibatches=8"
  # where between 4096 and 8192 does it break? 192*32, still divisible by 1024.
  "leading-6144|$BUFFERS_OFF|--ppo-override batch_size=192"
  # candidate configurations for the real runs: the largest env count the card
  # has been shown to step, with the leading dimension held at a passing 4096.
  "prod-4096|$BUFFERS_OFF|--num-envs 4096 --ppo-override num_minibatches=16"
  "prod-2048|$BUFFERS_OFF|--num-envs 2048 --ppo-override num_minibatches=16"
)

classify() {  # $1 = run dir, $2 = exit status
  local metrics="$1/metrics.jsonl"
  [[ "$2" != "0" ]] && { echo "CRASH-$2"; return; }
  [[ -s "$metrics" ]] || { echo "NO-METRICS"; return; }
  python3 - "$metrics" <<'PY'
import json, math, sys
rewards = []
for line in open(sys.argv[1]):
  row = json.loads(line)
  if "eval/episode_reward" in row:
    rewards.append(row["eval/episode_reward"])
if not rewards:
  print("NO-METRICS")
elif not all(math.isfinite(r) for r in rewards):
  print("NAN")
else:
  print("PASS" if rewards[-1] > 0.5 else "FLAT")
PY
}

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf "outcome\tlast_reward\tname\txla_flags\n" > "$SUMMARY"

for probe in "${PROBES[@]}"; do
  IFS='|' read -r name probe_flags extra_args <<< "$probe"
  if [[ ${#WANTED[@]} -gt 0 ]] && ! printf '%s\n' "${WANTED[@]}" | grep -qx "$name"; then
    continue
  fi
  flags="$probe_flags"
  run_dir="$OUT_DIR/$name"
  mkdir -p "$run_dir"
  echo "==> $name  [$flags] [$extra_args]"
  XLA_FLAGS="$flags" uv run --no-sync python benchmarks/train_playground_go1.py \
    --seed 0 --num-envs "$NUM_ENVS" --num-timesteps "$NUM_TIMESTEPS" \
    --out-dir "$run_dir" $extra_args >"$run_dir/log.txt" 2>&1
  status=$?
  outcome="$(classify "$run_dir" "$status")"
  last="$(tail -1 "$run_dir/metrics.jsonl" 2>/dev/null | python3 -c \
    'import json,sys; print(json.loads(sys.stdin.readline() or "{}").get("eval/episode_reward","-"))' 2>/dev/null || echo -)"
  printf "%s\t%s\t%s\t%s %s\n" "$outcome" "$last" "$name" "$flags" "$extra_args" >> "$SUMMARY"
  echo "    $outcome (last eval reward $last)"
done

echo
column -t -s $'\t' "$SUMMARY"
