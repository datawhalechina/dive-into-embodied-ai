"""Tracking metrics for a logged sim2sim run, per scripted command segment.

Commands are recomputed from the wireless log exactly as the deploy stack's
velocity_commands observation does (stick axes clamped to the deploy.yaml
ranges); actual velocity is the world-frame IMU-site velocity rotated into the
body frame by the logged base quaternion, with yaw rate taken from the gyro.
Error definitions match the training-side evaluation: mean L2 norm over the linear
command error and mean absolute angular command error.

Usage: python tracking_metrics.py <run_dir>  # writes <run_dir>/metrics.json
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from unitree_rl_mjx.export.deploy_config import go2_velocity_deploy_cfg

TRANSIENT_S = 2.0


def _load(path: Path, fields: list[str]) -> dict[str, np.ndarray]:
  if not path.exists():
    path = path.with_name(path.name + ".gz")
  columns = {f: [] for f in fields}
  with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
    for row in csv.DictReader(f):
      try:
        values = [float(row[field]) for field in fields]
      except (TypeError, ValueError):
        continue  # Partially flushed final row.
      for field, value in zip(fields, values):
        columns[field].append(value)
  return {f: np.asarray(v) for f, v in columns.items()}


def _nearest(t_from: np.ndarray, t_to: np.ndarray) -> np.ndarray:
  """Index into t_to of the nearest timestamp for each entry of t_from."""
  idx = np.searchsorted(t_to, t_from).clip(1, len(t_to) - 1)
  left = t_to[idx - 1]
  right = t_to[idx]
  return np.where(t_from - left < right - t_from, idx - 1, idx)


def _body_xy(quat: np.ndarray, v_world: np.ndarray) -> np.ndarray:
  """Rotate world-frame velocities into the body frame, keep x/y."""
  w, x, y, z = quat.T
  # Rows of R^T (world -> body) applied to v_world.
  vx = (
    (1 - 2 * (y * y + z * z)) * v_world[:, 0]
    + 2 * (x * y + w * z) * v_world[:, 1]
    + 2 * (x * z - w * y) * v_world[:, 2]
  )
  vy = (
    2 * (x * y - w * z) * v_world[:, 0]
    + (1 - 2 * (x * x + z * z)) * v_world[:, 1]
    + 2 * (y * z + w * x) * v_world[:, 2]
  )
  return np.stack([vx, vy], axis=1)


def main() -> None:
  run_dir = Path(sys.argv[1])
  segments = json.loads((run_dir / "segments.json").read_text())
  low = _load(run_dir / "lowstate.csv", ["t_ns", "qw", "qx", "qy", "qz", "wz"])
  high = _load(run_dir / "highstate.csv", ["t_ns", "vx", "vy", "vz"])
  wireless = _load(run_dir / "wireless.csv", ["t_ns", "lx", "ly", "rx"])

  ranges = go2_velocity_deploy_cfg()["commands"]["base_velocity"]["ranges"]
  cmd = np.stack(
    [
      np.clip(wireless["ly"], *ranges["lin_vel_x"]),
      np.clip(-wireless["lx"], *ranges["lin_vel_y"]),
      np.clip(-wireless["rx"], *ranges["ang_vel_z"]),
    ],
    axis=1,
  )

  quat = np.stack([low[k] for k in ("qw", "qx", "qy", "qz")], axis=1)
  v_world = np.stack(
    [high[k][_nearest(low["t_ns"], high["t_ns"])] for k in ("vx", "vy", "vz")],
    axis=1,
  )
  v_body = _body_xy(quat, v_world)
  cmd_at_low = cmd[_nearest(low["t_ns"], wireless["t_ns"])]

  results = []
  for seg in segments:
    duration = (seg["end_ns"] - seg["start_ns"]) / 1e9
    skip = min(TRANSIENT_S, duration / 2)
    lo_t = seg["start_ns"] + skip * 1e9
    mask = (low["t_ns"] >= lo_t) & (low["t_ns"] <= seg["end_ns"])
    lin_err = np.linalg.norm(cmd_at_low[mask, :2] - v_body[mask], axis=1)
    ang_err = np.abs(cmd_at_low[mask, 2] - low["wz"][mask])
    results.append(
      {
        "name": seg["name"],
        "command": [seg["vx"], seg["vy"], seg["wz"]],
        "steps": int(mask.sum()),
        "transient_skipped_s": skip,
        "mean_body_velocity": [
          float(v_body[mask, 0].mean()),
          float(v_body[mask, 1].mean()),
          float(low["wz"][mask].mean()),
        ],
        "lin_vel_error_mean": float(lin_err.mean()),
        "ang_vel_error_mean": float(ang_err.mean()),
      }
    )
    print(
      f"{seg['name']:8s} cmd=({seg['vx']:5.2f},{seg['vy']:5.2f},{seg['wz']:5.2f})"
      f"  actual=({results[-1]['mean_body_velocity'][0]:6.3f},"
      f"{results[-1]['mean_body_velocity'][1]:6.3f},"
      f"{results[-1]['mean_body_velocity'][2]:6.3f})"
      f"  lin_err={results[-1]['lin_vel_error_mean']:.4f}"
      f"  ang_err={results[-1]['ang_vel_error_mean']:.4f}"
    )

  moving = [r for r in results if r["name"] != "halt"]
  pooled = {
    "lin_vel_error_mean": float(
      np.mean([r["lin_vel_error_mean"] for r in moving])
    ),
    "ang_vel_error_mean": float(
      np.mean([r["ang_vel_error_mean"] for r in moving])
    ),
  }
  out = {"segments": results, "pooled_moving": pooled}
  (run_dir / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
  print(f"pooled (moving segments): lin_err={pooled['lin_vel_error_mean']:.4f}"
        f" ang_err={pooled['ang_vel_error_mean']:.4f}")
  print(f"wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
  main()
