# Unitree H1-2 (12-DoF) Web Simulation Bundle

Browser-ready MuJoCo scene and stateless LSTM ONNX walking policy for the **Unitree H1-2 (12-DoF)** full-size humanoid robot used by `/simulation`.

## Provenance & License

- **Upstream robot description & pretrained policy**: Unitree Robotics [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) (BSD 3-Clause License; see [LICENSE](LICENSE)).
- **Model configuration**: 12 active lower-body DoFs (`left/right_hip_yaw`, `hip_pitch`, `hip_roll`, `knee`, `ankle_pitch`, `ankle_roll`) with locked upper body and original foot contact geometry.
- **Mesh optimization**: The 54 binary STL visual meshes referenced by `h1_2_12dof.xml` were decimated from 39.37 MB to 1.66 MB while keeping original inertias, joint limits, actuator ranges, and contacts untouched (`provenance.json` records SHA-256 checksums).
- **Policy export**: Converted from the pretrained TorchScript LSTM checkpoint (`deploy/pre_train/h1_2/motion.pt`) into a stateless ONNX graph (`policy.onnx`, ~124 KB) accepting `obs (1, 47)`, `h_in (1, 1, 64)`, and `c_in (1, 1, 64)` and returning `action (1, 12)`, `h_out (1, 1, 64)`, and `c_out (1, 1, 64)`. Parity against the TorchScript model is verified to `9.54e-7` max absolute error across all fixtures in `reference.json`.

## Verification

From the repository root:

```bash
npm run check:simulation
```
