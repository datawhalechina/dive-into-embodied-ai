# CarDreamer intersection evaluation and future-BEV export

Reproduce **author public checkpoints**, without training a model. [Tutorial and videos](../../../../docs/foundations/world-model/3.cardreamer-right-turn.md). `experiment.py` is the current runner; `evaluate.py` and `results-clean4.json` retain the earlier debugging experiment, not the results below.

## Results

All three groups used separate fresh CARLA servers, Town03_Opt and seed 2024. Each group ran three consecutive episodes, not three independent seeds. All nine post-episode cleanup audits were empty. Different policies were not guaranteed identical traffic trajectories.

| Task | Checkpoint | Reached / episodes | Collisions | Forecast MAE | Frozen-frame MAE |
| --- | --- | --- | --- | --- | --- |
| `carla_right_turn_simple` | simple | 3/3 | 0 | 0.0251 | 0.0896 |
| `carla_right_turn_medium` | simple | 1/3 | 2 | 0.0512 | 0.0922 |
| `carla_right_turn_medium` | medium | 3/3 | 0 | 0.0337 | 0.0953 |

`results-advanced.json` includes all nine episode outcomes, cleanup audits and forecast averages. The medium task uses the author's 8–16 m traffic spawn spacing. Low-speed time below 0.2 m/s was 0.3–0.8 s including startup; this does not demonstrate reliable stopping or yielding. Minimum recorded distances are **vehicle-center distances**, not body clearance. Collision partner records establish contact, not fault.

MAE uses RGB values in 0–1, averaged over all nine predefined 20-frame windows per group, including collision episodes. Road pixels dominate the image. A lower pixel error does not establish accurate collision prediction. Both checkpoints contain a world model and policy, so this comparison does not isolate which component caused a driving difference.

## Tested setup and provenance

- [CarDreamer](https://github.com/ucd-dare/CarDreamer) commit `160132436aeda1de54956f9910e56f3970a565aa`; CARLA 0.9.15, Python 3.10.20.
- [Author checkpoints](https://huggingface.co/ucd-dare/CarDreamer/tree/main/CarDreamer_checkpoints):
  - `right_turn_simple.ckpt` SHA-256 `0a59b44b6c1433e5d36a979cf3ded1cbb8a0b4c0069cb1aa3d2f3192d00c784e`.
  - `right_turn_medium.ckpt` SHA-256 `7c2929263bf1617fb521a7193b672408cd413b382b189bbd2aea453d86802bb9`.
- RTX 4090, NVIDIA driver 535.98, Linux headless Vulkan. CUDA/cuDNN libraries were supplied separately. `requirements-eval.txt` lists Python packages, not a complete CUDA installation. Rendering saved predictions also needs the system `ffmpeg` executable.
- Town03_Opt replaces the author's default Town03 because the latter crashed in the UE4 Vulkan pipeline on this machine. Their OpenDRIVE files had identical SHA-256 `abf177bcd9b66cfefb7ac4919f6456815f1384f4229f1343c8a6a90bbb7713c9`; this does not establish complete scene equivalence.

## Preparation

Follow upstream installation, check out the pinned commit, and install compatible GPU libraries and Python packages. In that checkout run `git apply /path/to/eval-compat.patch`. The patch loads optional training replay modules lazily, reuses an already-loaded matching map and changes the client timeout from 20 to 60 seconds. It does not change policy actions or rewards. Optional training replay backends still need their dependencies.

In a **dedicated copy** of CARLA, back up `CarlaUE4/Config/DefaultEngine.ini` and replace `Town10HD_Opt.Town10HD_Opt` with `Town03_Opt.Town03_Opt` to start directly in the test map. Back up and move `CarlaUE4/Content/Carla/Maps/TM/Town03_Opt.bin` out of that path so the traffic manager builds its map in memory. The original cached file caused a reproducible traffic-manager crash in this setup; see [upstream issue #7210](https://github.com/carla-simulator/carla/issues/7210). These changes are specific to the tested setup and can be reverted afterwards.

Our offscreen server required removing inherited `SDL_VIDEODRIVER=offscreen` and configuring NVIDIA Vulkan/EGL driver discovery. Those driver paths depend on the machine. Start your **own dedicated server**, with no other users or simulation tasks, on unused ports:

```bash
env -u SDL_VIDEODRIVER ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=22000 -benchmark -fps=10
```

The runner refuses an initial scene containing vehicles or sensors. It destroys only actors owned by its task, applies one world tick, and requires an empty scene after each episode. Do not use it on a shared CARLA world. The runner does not start or terminate the server; stop only the server process you started.

## Run the three groups

Run each command from this directory in the prepared Python environment. **Stop your dedicated server and start a fresh one between groups**; use a new output directory for every experiment. Checkpoints must come from a trusted source.

```bash
# 1. Empty road; simple checkpoint.
python experiment.py --source /path/to/CarDreamer \
  --checkpoint /path/to/right_turn_simple.ckpt --output results/empty \
  --port 22000 --task carla_right_turn_simple --episodes 3 --seed 2024
```

```bash
# 2. Restart your server first. Add traffic; retain simple checkpoint.
python experiment.py --source /path/to/CarDreamer \
  --checkpoint /path/to/right_turn_simple.ckpt --output results/transfer \
  --port 22000 --task carla_right_turn_medium --episodes 3 --seed 2024
```

```bash
# 3. Restart your server first. Same traffic configuration; medium checkpoint.
python experiment.py --source /path/to/CarDreamer \
  --checkpoint /path/to/right_turn_medium.ckpt --output results/matched \
  --port 22000 --task carla_right_turn_medium --episodes 3 --seed 2024
```

The runner disables the optional HTTP monitor. It records terminal flags, observed collisions and a separate collision sensor. Success requires arrival, no collision from any of these sources, no lane departure, no environment timeout and no evaluator truncation. Arrival means at most three route waypoints remain; lane departure is waypoint-distance based. The upstream collision observation can lag by one frame. These criteria do not establish real-world safety.

Each episode exports:

- `result.json`, `steps.json`, `collisions.json`: outcomes, controls, motion and contact partners; partial steps are saved during execution.
- `replay.mp4`: 960×540 chase camera plus actual simulator BEV. It is **not a model prediction**. Frame IDs pair camera and BEV; missing camera frames are counted and omitted (3–10 per episode in this run).
- `scene-start.json`, `scene-after-cleanup.json`: actor inventories; a nonempty cleanup audit stops the experiment.
- `trajectory.npz`: actual BEV observations, recorded actions and simulation frame IDs.
- `forecast-020.npz`, `forecast-060.npz`, `forecast-100.npz`, `forecasts.json`: future predictions, truth, context, actions and MAE. Windows beyond early termination are skipped, rather than replaced with different windows.

## Forecast alignment and visualization

At observation index `t = 20, 60, 100` (2, 6, 10 simulated seconds), the agent has filtered observations through `obs[t]`. Starting from that latent state, RSSM receives the **recorded actions** `a[t:t+20]` and predicts `obs[t+1:t+21]`, a two-second horizon. No future image enters the forecast function. One stochastic sample is generated with fixed key `42+t` for each window; uncertainty is not estimated. This is offline action-conditioned forecasting, not online MPC or independent prediction of future policy actions.

```bash
python render_forecasts.py results/empty/episode-00 --output prediction-empty.mp4
python render_forecasts.py results/matched/episode-00 --output prediction-traffic.mp4
```

The renderer only needs CPU, NumPy, OpenCV and FFmpeg. Three panels show actual future BEV, decoded prediction and absolute RGB error averaged across channels and amplified 4×. Native BEV is 128×128, enlarged for viewing. Original steps are 10 Hz; videos play at 5 fps (0.5×), for 12 seconds across three windows. The frozen-frame baseline repeats `obs[t]` across the horizon.

## Earlier debugging results

The previous evaluator recorded four arrivals followed by a communication timeout. Another batch reusing an old scene had collisions with scene contamination unresolved. Both are excluded from the nine-episode comparison above. The new 60-second client timeout and cleanup audits allowed this batch to finish, but do not prove the earlier timeout's root cause has been resolved. All current failures remain in `results-advanced.json`.
