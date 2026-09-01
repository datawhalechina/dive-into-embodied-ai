"""Re-render a logged sim2sim run with a tracking camera.

The session screen capture uses simulate's free camera, which the robot walks
out of; this reconstructs the same trajectory from the logged base pose and
joint positions and renders it with a camera that follows the trunk. Frames
are piped straight to ffmpeg.

Usage: python render_from_logs.py <run_dir> [out.mp4]
"""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE = (
  Path(__file__).resolve().parents[2]
  / "src/unitree_rl_mjx/assets/robots/unitree_go2/xmls/scene_go2.xml"
)
# Lowstate motor order (bridge sensor order): FR, FL, RR, RL triplets.
JOINT_NAMES = [
  f"{leg}_{part}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for part in ("hip", "thigh", "calf")
]
FPS = 30
SIZE = (1280, 720)


def _load(path: Path, fields: list[str]) -> dict[str, np.ndarray]:
  if not path.exists():
    path = path.with_name(path.name + ".gz")
  columns = {f: [] for f in fields}
  with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
    for row in csv.DictReader(f):
      try:
        values = [float(row[field]) for field in fields]
      except (TypeError, ValueError):
        continue
      for field, value in zip(fields, values):
        columns[field].append(value)
  return {f: np.asarray(v) for f, v in columns.items()}


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  rot = np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )
  return rot @ vec


def main() -> None:
  run_dir = Path(sys.argv[1])
  out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "tracked.mp4"
  segments = json.loads((run_dir / "segments.json").read_text())
  low_fields = ["t_ns", "qw", "qx", "qy", "qz"] + [f"q{i}" for i in range(12)]
  low = _load(run_dir / "lowstate.csv", low_fields)
  high = _load(run_dir / "highstate.csv", ["t_ns", "px", "py", "pz"])

  model = mujoco.MjModel.from_xml_path(str(SCENE))
  model.vis.global_.offwidth = SIZE[0]
  model.vis.global_.offheight = SIZE[1]
  data = mujoco.MjData(model)
  qpos_adr = [
    model.joint(name).qposadr[0] for name in JOINT_NAMES
  ]
  imu_site = model.site("imu")
  site_offset = imu_site.pos.copy()

  # Start shortly before the FixStand chord (12 s before the first segment)
  # rather than at logger start, which precedes the driver by a long idle.
  t0 = max(low["t_ns"][0], segments[0]["start_ns"] - 12_000_000_000)
  t1 = segments[-1]["end_ns"] + 2_000_000_000
  grid = np.arange(t0, t1, 1e9 / FPS)
  low_idx = np.searchsorted(low["t_ns"], grid).clip(0, len(low["t_ns"]) - 1)
  high_idx = np.searchsorted(high["t_ns"], grid).clip(0, len(high["t_ns"]) - 1)

  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  camera.trackbodyid = imu_site.bodyid[0]
  camera.distance = 1.6
  camera.elevation = -18
  camera.azimuth = 125

  ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-video_size", f"{SIZE[0]}x{SIZE[1]}", "-framerate", str(FPS), "-i", "-",
     "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
     str(out_path)],
    stdin=subprocess.PIPE,
  )
  renderer = mujoco.Renderer(model, height=SIZE[1], width=SIZE[0])
  for li, hi in zip(low_idx, high_idx):
    quat = np.array([low[k][li] for k in ("qw", "qx", "qy", "qz")])
    p_site = np.array([high[k][hi] for k in ("px", "py", "pz")])
    data.qpos[0:3] = p_site - _quat_rotate(quat, site_offset)
    data.qpos[3:7] = quat
    for j, adr in enumerate(qpos_adr):
      data.qpos[adr] = low[f"q{j}"][li]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera)
    ffmpeg.stdin.write(renderer.render().tobytes())
  ffmpeg.stdin.close()
  ffmpeg.wait()
  renderer.close()
  print(f"wrote {out_path} ({len(grid)} frames)")


if __name__ == "__main__":
  main()
