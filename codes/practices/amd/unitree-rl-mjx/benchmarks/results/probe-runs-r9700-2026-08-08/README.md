# Raw probe runs from the ROCm box (2026-08-08 investigation)

Working artifacts from the gfx1201 sort-defect investigation: logs, per-epoch
metrics, and run manifests. Checkpoints and trajectories are omitted (large
and reproducible).

The summary these runs fed:

- `benchmarks/results/2026-08-08-sort-defect-rocm-r9700.json` — the probe
  summary

The summary states findings; these raw runs are the evidence behind them —
the band boundaries and XLA-flag conclusions can be re-derived from what is
here.

Contents: eight XLA-flag variants (`probe-2026-08-08/`), env-count threshold
probes (`probe-threshold-*`), batch-shape probes (`probe-shapes-*`),
production-config checks (`probe-prod-*`), and the post-fix verification runs
(`sortfix-*`).
