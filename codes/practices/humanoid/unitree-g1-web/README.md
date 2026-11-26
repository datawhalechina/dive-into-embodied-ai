# Unitree G1 (12-DoF) Web Simulation Bundle

Browser-ready MuJoCo scene and stateless LSTM ONNX walking policy for the **Unitree G1 (12-DoF)** humanoid robot used by `/simulation`.

## Provenance & License

- **Upstream robot description & pretrained policy**: Unitree Robotics [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) (BSD 3-Clause License; see [LICENSE](LICENSE)).
- **Model configuration**: 12 active lower-body DoFs (`left/right_hip_pitch`, `hip_roll`, `hip_yaw`, `knee`, `ankle_pitch`, `ankle_roll`) with locked upper body and original 4-sphere-per-foot contact primitives.
- **Mesh optimization**: The 27 binary STL visual meshes were decimated from 24.0 MB to 1.49 MB while keeping original inertias, joint limits, actuator ranges, and contact primitives untouched (`provenance.json` records SHA-256 checksums).
- **Policy export**: Converted from the pretrained TorchScript LSTM checkpoint (`policy_lstm_1.pt`) into a stateless ONNX graph (`policy.onnx`, ~127 KB) accepting `obs (1, 47)`, `h_in (1, 1, 64)`, and `c_in (1, 1, 64)` and returning `action (1, 12)`, `h_out (1, 1, 64)`, and `c_out (1, 1, 64)`. Parity against the TorchScript model is verified to `< 2e-6` max absolute error across all fixtures in `reference.json`.

## Verification

From the repository root:

```bash
npm run check:simulation
```
