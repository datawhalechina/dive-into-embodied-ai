# Exported policy: Go2 flat velocity, v0

- Source: `benchmarks/results/train-go2-r9700-full-seed1/params.bin` (trained
  on a Radeon AI PRO R9700; best held-out linear tracking of the three
  full-run seeds, 0.216 m/s).
- Contract verification at export: max |JAX − ONNX| = 1.67e-06 over 256
  observations drawn from the policy's running statistics (tolerance 1e-5);
  `tests/test_policy_export.py` re-verifies on random and recorded
  observations.
- Install into an upstream checkout:

      python -m unitree_rl_mjx.export.install --robot go2 \
        --params benchmarks/results/train-go2-r9700-full-seed1/params.bin \
        --dest <checkout> --version v0

> 注：来源 checkpoint（`params.bin`）未随仓库分发；上面的验证数字是导出时
> 的记录，`exported/policy.onnx` 与 `params/deploy.yaml` 即该次导出的产物。
