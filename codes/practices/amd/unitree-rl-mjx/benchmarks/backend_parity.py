"""Compare backend seed groups: plateau bands, held-out metrics, verdict.

Reads full-run directories (metrics.jsonl + eval.json each) grouped by
backend, computes plateau statistics over the final fraction of training,
tabulates the held-out evaluation metrics per seed, and prints an explicit
accept/reject against the parity bar: plateau bands overlap AND held-out
metrics interleave (no metric where every seed of one backend falls outside
the other backend's range).

    uv run python benchmarks/backend_parity.py \
        "rocm=r9700-s0,r9700-s1,r9700-s2" "cuda=a40-s0,a40-s1,a40-s2"
"""

import argparse
import json
from pathlib import Path

REWARD_KEY = "eval/episode_reward"
PLATEAU_FRACTION = 0.10
METRICS = (
  "lin_vel_error_mean",
  "ang_vel_error_mean",
  "survival_rate",
  "air_time_cv",
  "episode_return_mean",
)


def _plateau_values(run_dir: Path) -> list[float]:
  rows = [
    json.loads(line)
    for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    if line.strip()
  ]
  rows = [r for r in rows if REWARD_KEY in r]
  cutoff = rows[-1]["step"] * (1.0 - PLATEAU_FRACTION)
  return [r[REWARD_KEY] for r in rows if r["step"] >= cutoff]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("backends", nargs=2, help="LABEL=run_dir,run_dir,...")
  a = p.parse_args()

  groups = {}
  for spec in a.backends:
    label, _, paths = spec.partition("=")
    groups[label] = [Path(path) for path in paths.split(",")]

  print(f"== plateau (final {PLATEAU_FRACTION:.0%} of steps, per-seed evals pooled)")
  bands = {}
  for label, dirs in groups.items():
    pooled = [v for d in dirs for v in _plateau_values(d)]
    bands[label] = (min(pooled), max(pooled))
    mean = sum(pooled) / len(pooled)
    print(
      f"  {label:>6}: mean {mean:6.2f}   band [{bands[label][0]:6.2f}, "
      f"{bands[label][1]:6.2f}]   ({len(pooled)} evals over {len(dirs)} seeds)"
    )
  (lo_a, hi_a), (lo_b, hi_b) = bands.values()
  bands_overlap = lo_a <= hi_b and lo_b <= hi_a
  print(f"  bands overlap: {bands_overlap}")

  print("\n== held-out metrics per seed")
  evals = {
    label: [json.loads((d / "eval.json").read_text())["metrics"] for d in dirs]
    for label, dirs in groups.items()
  }
  labels = list(groups)
  header = (
    "  "
    + f"{'metric':>22}"
    + "".join(f"   {label}[{i}]" for label in labels for i in range(len(groups[label])))
  )
  print(header)
  interleaved = {}
  for m in METRICS:
    row = f"  {m:>22}"
    for label in labels:
      for e in evals[label]:
        row += f"   {e[m]:7.4f}"
    a_vals = [e[m] for e in evals[labels[0]]]
    b_vals = [e[m] for e in evals[labels[1]]]
    disjoint = min(a_vals) > max(b_vals) or max(a_vals) < min(b_vals)
    interleaved[m] = not disjoint
    row += "   " + ("interleaved" if not disjoint else "DISJOINT")
    print(row)

  print("\n== verdict")
  all_interleaved = all(interleaved.values())
  accept = bands_overlap and all_interleaved
  print(f"  plateau bands overlap:      {bands_overlap}")
  print(f"  all metrics interleaved:    {all_interleaved}")
  for m, ok in interleaved.items():
    if not ok:
      print(f"    disjoint metric: {m}")
  print(f"  PARITY: {'ACCEPT' if accept else 'REJECT'}")


if __name__ == "__main__":
  main()
