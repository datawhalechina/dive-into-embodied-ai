"""Decompose the full runs' wall-clock into training, eval, and residual.

Reads the committed metrics.jsonl + run.json of the six full training runs.
brax reports cumulative training/eval walltimes per epoch, so the split is:
training epochs (first-epoch delta carries XLA compilation), the periodic
evaluator, and a residual (env construction, reset compilation, checkpoint
save, trajectory rollout). Steady-state throughput uses the median epoch.

    uv run python benchmarks/decompose_walltime.py  # writes JSON next to it
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
RUNS = [
  f"train-go2-{box}-full-seed{seed}"
  for box in ("a40", "r9700")
  for seed in (0, 1, 2)
]
OUT = RESULTS / "2026-08-10-walltime-decomposition.json"


def decompose(run_dir: Path) -> dict:
  run = json.loads((run_dir / "run.json").read_text())
  rows = [
    json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
  ]
  train_rows = [r for r in rows if "training/walltime" in r]
  steps = np.array([r["step"] for r in train_rows])
  t_train = np.array([r["training/walltime"] for r in train_rows])
  t_eval = np.array([r["eval/walltime"] for r in rows if "eval/walltime" in r])
  epoch_s = np.diff(np.concatenate([[0.0], t_train]))
  epoch_steps = np.diff(np.concatenate([[0], steps]))
  median_epoch_s = float(np.median(epoch_s[1:]))
  return {
    "run": run_dir.name,
    "device_kind": run["device_kind"],
    "wall_s": run["wall_s"],
    "training_s": float(t_train[-1]),
    "eval_s": float(t_eval[-1]),
    "residual_s": float(run["wall_s"] - t_train[-1] - t_eval[-1]),
    "epochs": len(epoch_s),
    "first_epoch_s": float(epoch_s[0]),
    "median_epoch_s": median_epoch_s,
    # First-epoch excess over the median epoch; the training-step compile
    # happens inside the first epoch, so this is an estimate of it.
    "est_train_compile_s": float(epoch_s[0] - median_epoch_s),
    "steady_sps_median": float(np.median(epoch_steps[1:] / epoch_s[1:])),
    "e2e_sps": float(steps[-1] / run["wall_s"]),
  }


def main() -> None:
  runs = [decompose(RESULTS / name) for name in RUNS]
  by_device = {}
  for r in runs:
    by_device.setdefault(r["device_kind"], []).append(r)
  summary = {}
  for device, rs in by_device.items():
    summary[device] = {
      "mean_wall_s": float(np.mean([r["wall_s"] for r in rs])),
      "mean_training_s": float(np.mean([r["training_s"] for r in rs])),
      "mean_eval_s": float(np.mean([r["eval_s"] for r in rs])),
      "mean_residual_s": float(np.mean([r["residual_s"] for r in rs])),
      "mean_steady_sps": float(np.mean([r["steady_sps_median"] for r in rs])),
      "mean_est_train_compile_s": float(
        np.mean([r["est_train_compile_s"] for r in rs])
      ),
    }
  devices = list(summary)
  ratios = {
    "wall": summary[devices[1]]["mean_wall_s"] / summary[devices[0]]["mean_wall_s"],
    "training": summary[devices[1]]["mean_training_s"]
    / summary[devices[0]]["mean_training_s"],
    "steady_sps": summary[devices[0]]["mean_steady_sps"]
    / summary[devices[1]]["mean_steady_sps"],
  }
  payload = {"runs": runs, "summary": summary, "ratios_second_over_first": ratios}

  # Rollout vs learner split at the training batch size, bounded by the bare
  # env.step ladders: per-step time not spent in env.step is the learner
  # update plus training-wrapper overhead (resets, reshuffles) the bare
  # ladder does not pay.
  split = {}
  for device, ladder in (
    ("NVIDIA A40", "2026-08-10-go2-envstep-cuda-a40.json"),
    ("AMD Radeon AI PRO R9700", "2026-08-10-go2-envstep-rocm-r9700.json"),
  ):
    rows = json.loads((RESULTS / ladder).read_text())["results"]
    env_sps = next(r["steps_per_s"] for r in rows if r["num_envs"] == 4096)
    step_us = 1e6 / summary[device]["mean_steady_sps"]
    rollout_us = 1e6 / env_sps
    split[device] = {
      "env_step_sps_4096": env_sps,
      "step_us": step_us,
      "rollout_us": rollout_us,
      "learner_and_wrappers_us": step_us - rollout_us,
      "rollout_share": rollout_us / step_us,
    }
    print(
      f"{device}: step {step_us:.1f}us = rollout {rollout_us:.1f}us"
      f" ({100 * rollout_us / step_us:.0f}%) + learner/wrappers"
      f" {step_us - rollout_us:.1f}us"
    )
  payload["rollout_update_split_4096"] = split
  OUT.write_text(json.dumps(payload, indent=2) + "\n")
  for device, s in summary.items():
    print(
      f"{device}: wall {s['mean_wall_s']:.0f}s = training {s['mean_training_s']:.0f}s"
      f" + eval {s['mean_eval_s']:.0f}s + residual {s['mean_residual_s']:.0f}s;"
      f" steady {s['mean_steady_sps']:,.0f} sps;"
      f" est. train compile {s['mean_est_train_compile_s']:.0f}s"
    )
  print(f"ratios ({devices[1]} / {devices[0]}): {json.dumps(ratios)}")
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
