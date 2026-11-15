# Online simulation model library

[中文](README.md) · [Italiano](README.it.md)

The `/simulation` route shares one catalog across Chinese, English, and Italian. **Free control** turns keyboard and touch input into movement commands. **Policy showcase** supplies the model's `demoCommand` automatically. Physics and policy inference run in a browser Worker; there is no remote training service.

## Included models

| Model | Controller | Movement |
| --- | --- | --- |
| Unitree Go2 | Repository PPO v0 ONNX weights, 50 Hz inference | Forward, reverse, turning, strafing |
| Pupper v3 | Project 8M-step PPO checkpoint, 50 Hz inference | Forward, reverse, turning, strafing |
| MicroDuck | 14 position servos, no trained weights | Head keys, joint sliders, base support toggle |
| Petoi Quaddle | Existing periodic gait, no trained weights | Forward, walking turns, stop |

Go2 uses 47 observations in training order FL / FR / RL / RR: body angular velocity, projected gravity, commands, gait phase, joint state, and previous actions. Normalization is embedded in ONNX. Actions become joint targets and PD torques, mapped by joint name to actuator order. Stop sends zero velocity; pause freezes physics.

The original `scene_go2.xml` retains inertias, joints, actuators, and collision geometry. The build removes visual meshes that require a separate download and displays collision shapes instead. Provenance, weight hashes, and licenses ship with the assets. Browser checks do not validate physical hardware or training quality.

## Pupper v3 and MicroDuck

Pupper v3 uses the archived 8M-step PPO checkpoint in `cs123/6.rl_pupper_v2_5` with original meshes. Its 53 observations follow FR / FL / RR / RL ordering, including the training coordinate transforms and trot phase. Physics runs at 4 ms, control at 20 ms, with action scale 0.5, EMA filtering, and KP=5 / KD=0.25. A zero command still allows some drift.

The practice project's `web/` folder contains ONNX, configuration, native reference samples, and source hashes. To regenerate, install `onnx==1.23.2` and `onnxruntime==1.31.1` in the cs123 SB3/PyTorch environment, then run `codes/practices/quadruped/cs123/.venv/bin/python scripts/export-pupper-web.py` from the repository root. The exporter compares 256 deterministic SB3 and ONNX actions. Website builds use the saved export without Python or training dependencies.

MicroDuck uses original meshes, 14 position servos, and the STAND pose. No trained weights are connected, so only free control is enabled. W/S pitch the head, A/D turn it, and Q/E roll it; sliders control all joints. A weld holds the base by default. Release it for free physics; enabling it again or resetting restores the initial pose. These are native MJCF servos, not the training BAM actuator model. Upstream code license and mesh license notice ship with the assets.

## Adding trained models

1. Store distributable model assets, weights, inference configuration, and licenses with the practice project. Export browser-compatible ONNX; `.pt`, `.pth`, and SB3 `.zip` files cannot run directly.
2. Register an ID, name, `kind`, `adapter`, source paths, tracking body, camera distance, demo command, and tutorial in `models.json`. Another policy using the same Go2 contract can reuse its adapter.
3. For a different robot or policy, add a `runtime/` adapter and connect it in `worker.mjs` and `scripts/simulation-assets.cjs`. Specify observation order and frames, history, normalization, action scaling, joint order, timing, initialization, and reset. An ONNX file alone does not define these. The inference adapters support the known Go2 and Pupper contracts, not arbitrary uploaded weights.
4. Update model descriptions in all three languages. Edit source files only; `static/simulation-assets/` and `static/quaddle/` are generated. Do not fetch unpinned models or execute user scripts during builds.
5. Add real physics/inference checks for movement, stopping, reset, and invalid outputs. Run `check:simulation`, `check:quaddle-web`, `typecheck`, `assets:webp:check`, `build`, `check:i18n`, and `check:page-markdown` with npm.

Browser checks should cover keyboard and touch, mode/model switching, release, focus loss, hidden pages, pause/reset, repeated loading, error recovery, narrow screens, and both color themes. Shortcuts only apply when the scene has focus; leaving clears commands. Slower devices may run below real time without catching up after background suspension.
