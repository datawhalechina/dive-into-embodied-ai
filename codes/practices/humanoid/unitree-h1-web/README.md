# Unitree H1 (10-DoF) Web Simulation Bundle

Browser-ready MuJoCo scene and stateless LSTM ONNX walking policy for the **Unitree H1 (10-DoF)** full-size humanoid robot used by `/simulation`.

## Provenance & License

- **Upstream robot description & pretrained policy**: Unitree Robotics [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) (BSD 3-Clause License; see [LICENSE](LICENSE)).
- **Model configuration**: 10 active lower-body DoFs (`left/right_hip_yaw`, `hip_roll`, `hip_pitch`, `knee`, `ankle`) with locked upper body and original foot contact geometry.
- **Mesh optimization**: The 21 binary STL visual meshes were decimated from 14.15 MB to 1.02 MB while keeping original inertias, joint limits, actuator ranges, and contacts untouched (`provenance.json` records SHA-256 checksums).
- **Policy export**: Converted from the pretrained TorchScript LSTM checkpoint (`deploy/pre_train/h1/motion.pt`) into a stateless ONNX graph (`policy.onnx`, ~118 KB) accepting `obs (1, 41)`, `h_in (1, 1, 64)`, and `c_in (1, 1, 64)` and returning `action (1, 10)`, `h_out (1, 1, 64)`, and `c_out (1, 1, 64)`. Parity against the TorchScript model is verified to `1.43e-6` max absolute error across all fixtures in `reference.json`.

## Verification

From the repository root:

```bash
npm run check:simulation
```
