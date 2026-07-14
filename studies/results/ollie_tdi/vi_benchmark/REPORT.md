# Mojito VI warm-start benchmark

Last generated: 2026-07-13T22:18:12.050968+00:00

## Decision rule (declared before running)

VI becomes the default only if the paired MCMC posteriors pass the equivalence gates and VI either reduces total estimator wall time by at least 10% or materially improves sampling diagnostics. If VI merely adds at least 5% cost with no material diagnostic gain, it is a candidate for removal after this report and the JSON/NPZ artifacts are retained.

Posterior equivalence requires pointwise-normalized log-surface RMS <= 0.25, at most 1% of cells beyond one pooled posterior standard deviation, and each phi posterior mean shift <= 0.5 pooled posterior standard deviations.

## Method

The benchmark uses channel X, start day 20, band 1e-4--0.1 Hz, seed 0, `centered=True`, 30 interior frequency knots, and the exact `1_week` and `1_month` settings in `mojito_experiments.EXPERIMENTS` (300 warmup + 300 posterior draws in each of two sequential chains). The VI arm uses the current defaults: 2,000 AutoDiagonalNormal SVI steps at learning rate 1e-2. Each arm runs in a fresh Python process so compiled JAX state is not shared.

The thresholds and exact case settings were frozen before results in `PROTOCOL.json`; completed arms additionally record their script, inference, and VI source hashes.

Total estimator time covers WDM transformation, basis/PLS setup, optional VI, NUTS, and surface reconstruction, but excludes HDF5 loading. Tree-depth saturation means `num_steps >= 2^10 - 1`. The 90% log-surface interval is converted to an approximate posterior standard deviation for paired-surface normalization.

## Exact launch commands

```bash
uv run python studies/ollie_tdi/benchmark_vi_warmstart.py --preflight
uv run python studies/ollie_tdi/benchmark_vi_warmstart.py
uv run python studies/ollie_tdi/benchmark_vi_warmstart.py --report-only
```

If the exact file is mounted elsewhere, set
`MOJITO_DATA=/absolute/path/to/processed_segments_noise_no_segmentation.h5`.

Completed outputs are preserved on rerun. Use `--force` only to deliberately replace an existing arm.

## Results

Benchmark incomplete. Missing paired artifacts for: `1_week`, `1_month`.

Required external input: `/Users/avi/Documents/projects/MojitoProcessor/Mojito_Data/processed_segments_noise_no_segmentation.h5`.

## Overall conclusion

No decision yet. Do not change the default or delete VI until both paired Mojito cases complete.
