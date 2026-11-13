"""诊断 1（快速版）：直接读 parquet 扫动作分布（不解码视频）。"""
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd

ROOT = "/workspace/dive-into-embodied-ai/codes/practices/quadruped/quaddle-mujoco/outputs/dataset_merged"
files = sorted(glob.glob(f"{ROOT}/data/chunk-*/file-*.parquet"))
print(f"parquet files: {len(files)}")

dfs = [pd.read_parquet(f, columns=["action", "observation.state"]) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"total frames: {len(df)}")

acts = np.stack(df["action"].to_numpy())
states = np.stack(df["observation.state"].to_numpy())
vx, wz = acts[:, 0], acts[:, 1]
bearings = states[:, 6]

cnt = Counter()
by_zone = Counter()
for v, w, b in zip(vx, wz, bearings):
    if abs(w) >= abs(v) and abs(w) > 0.15:
        k = "steer_L" if w > 0 else "steer_R"
    elif v > 0 and abs(v) >= 0.15:
        k = "forward"
    else:
        k = "stop"
    cnt[k] += 1
    zone = "rear>120" if abs(b) > 2.094 else ("mid 60-120" if abs(b) > 1.047 else "front<60")
    by_zone[(zone, k)] += 1

total = len(df)
print("\n== 帧级动作分布 ==")
for k, v in cnt.most_common():
    print(f"  {k:9s} {v:7d}  {v/total*100:5.1f}%")
print(f"\nwz: [{wz.min():.2f},{wz.max():.2f}] 均值 {wz.mean():.3f} | vx: [{vx.min():.2f},{vx.max():.2f}] 均值 {vx.mean():.3f}")

print("\n== 按 bearing 区域 ==")
for zone in ("front<60", "mid 60-120", "rear>120"):
    row = {k: c for (z, k), c in by_zone.items() if z == zone}
    zt = sum(row.values())
    if zt:
        print(f"  {zone:12s} n={zt:6d}  " + "  ".join(f"{k}:{v/zt*100:.1f}%" for k, v in sorted(row.items(), key=lambda x: -x[1])))
os._exit(0)
