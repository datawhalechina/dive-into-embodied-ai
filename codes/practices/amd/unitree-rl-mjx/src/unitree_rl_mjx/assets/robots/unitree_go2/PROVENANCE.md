# Go2 robot description — provenance

The MJCF files under `xmls/` are copied unmodified from Unitree's
`unitree_rl_mjlab` repository, which in turn derives them from MuJoCo Menagerie.

| | |
|---|---|
| Source | https://github.com/unitreerobotics/unitree_rl_mjlab |
| Path | `src/assets/robots/unitree_go2/xmls/` |
| Commit | `1425b15f73bd4095f0df53709d7c389c3eb9e790` (2026-04-13) |
| Upstream license | see `LICENSE-mjlab` in this directory |
| Meshes | 16 `.obj` files, fetched into `xmls/assets/` by `scripts/fetch_go2_meshes.sh`（SHA-256 校验，同一提交） |

SHA-256 of the copied MJCF files, for verifying they still match the source:

```
077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912  go2.xml
97c7ece37cb37666f2767abb4449c3a8a6a2a0e90716d14940e09423f5d426ef  scene_go2.xml
```

Verify with `shasum -a 256 xmls/go2.xml xmls/scene_go2.xml`.

## The two files are not interchangeable

- `go2.xml` — the robot alone: 12 leg joints plus a free joint, collision and visual
  geoms, foot sites, IMU site. It has **no actuators, no keyframe and no ground
  plane**; an environment supplies those. This is the file training uses.
- `scene_go2.xml` — a standalone scene: same robot with torque motors, a ground
  plane, and a `home` keyframe. Used for visualization and sim2sim.

Their rest poses differ, which is expected and load-bearing: `scene_go2.xml`'s `home`
keyframe puts the base at z = 0.27 with hips at 0, while training starts from
`INIT_STATE` (z = 0.32, hips ±0.1). Training and deployment both use the latter.
