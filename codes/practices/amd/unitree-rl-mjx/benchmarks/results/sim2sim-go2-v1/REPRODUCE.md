# Sim2sim closed loop: Go2 velocity policy through the official C++ stack

Run 2026-08-10 on a40-box (Ubuntu 22.04, CPU-only workload),
following the procedure in `docs/tutorials/03-sim2sim.md`.

## What ran

The unmodified vendored stack: `simulate/` (MuJoCo 3.3.6 bridge, go2 scene)
and `deploy/robots/go2` (`go2_ctrl`), over DDS loopback, headless under Xvfb.
The policy is `v1`, exported on the box by

    uv run python -m unitree_rl_mjx.export.install --robot go2 \
      --params benchmarks/results/train-go2-r9700-full-seed1/params.bin \
      --dest <unitree_rl_mjlab 检出目录> --version v1

(same source checkpoint as the shipped `policies/go2/velocity/v0`; export
verification on the box: max |JAX − ONNX| = 2.15e-06). FSM transitions and
command segments were driven by `benchmarks/sim2sim/js_driver.py` through a
FIFO virtual gamepad; `benchmarks/sim2sim/logger/` recorded the DDS topics.

FSM log: Passive → FixStand 10:57:13, FixStand → Velocity 10:57:19,
Velocity → Passive 10:57:51 (clean exit; no crashes).

## Files

- `lowstate.csv.gz` / `highstate.csv.gz` / `wireless.csv.gz` — full-rate DDS
  logs (base quat + gyro + joints / world-frame IMU-site pos + vel / stick
  axes).
- `segments.json` — scripted command segments with wall-clock boundaries.
- `metrics.json` — per-segment tracking metrics from
  `benchmarks/sim2sim/tracking_metrics.py` (same error definitions as the training-side evaluation).
- `session.mp4` — raw x11grab of the simulate GUI (free camera; the robot
  walks out of view during the forward segment).
- `tracked.mp4` — the same logged trajectory re-rendered with a tracking
  camera by `benchmarks/sim2sim/render_from_logs.py` (reconstruction from the
  logs above, not a separate simulation).

## Numbers

| Segment | Command (vx, vy, wz) | Mean actual | lin_err | ang_err |
|---|---|---|---|---|
| forward | (1.0, 0, 0) | (0.847, −0.009, −0.007) | 0.161 | 0.048 |
| lateral | (0, 0.4, 0) | (0.014, 0.182, 0.003) | 0.219 | 0.019 |
| turn | (0, 0, 0.8) | (−0.004, 0.009, 0.410) | 0.035 | 0.390 |
| halt (×3) | (0, 0, 0) | ≈ 0 | ≤ 0.001 | ≤ 0.001 |

Means over each segment excluding a 2 s transient (1 s for the 2 s halts).

`mjx_reference.json` (from `benchmarks/sim2sim/mjx_reference.py`) rolls the
same checkpoint in the training environment under the same three commands,
pushed through the env's heading-servo command interface:

| Segment | MJX actual | sim2sim actual |
|---|---|---|
| forward vx=1.0 | 0.930 | 0.847 |
| lateral vy=0.4 | 0.296 | 0.182 |
| turn wz=0.8 | 0.805 | 0.410 |

Forward carries over well. Lateral undershoot is partly the policy's own
(0.296 already in MJX) and partly the sim gap. Yaw tracks essentially
perfectly in MJX under identical observations, so the sim2sim shortfall to
0.410 is a genuine simulator/model gap (full-collision official scene,
different solver settings, 1 kHz PD path), not policy incapacity —
quantified here.

> 注：本目录只随仓库分发小体积证据（json/jsonl/npz/csv/md）。上文提到的
> 训练 checkpoint（`params.bin`）、完整 DDS 日志（`*.csv.gz`）、视频（`*.mp4`）
> 与已执行 notebook 不在其中，可按本文步骤复现生成。官方 simulate/deploy 栈
> 是独立检出（见 `docs/tutorials/03-sim2sim.md`），不在本目录内。
