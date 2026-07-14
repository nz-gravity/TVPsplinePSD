# Ollie 30-day proxy: VI warm-start benchmark

Last generated: 2026-07-13T22:30:41.543782+00:00

## Decision rule (declared before running)

VI becomes the default only if the paired MCMC posteriors pass the equivalence gates and VI either reduces total estimator wall time by at least 10% or materially improves sampling diagnostics. If VI merely adds at least 5% cost with no material diagnostic gain, it is a candidate for removal after this report and the JSON/NPZ artifacts are retained.

Posterior equivalence requires pointwise-normalized log-surface RMS <= 0.25, at most 1% of cells beyond one pooled posterior standard deviation, and each phi posterior mean shift <= 0.5 pooled posterior standard deviations.

## Method

This is explicitly proxy evidence, not the unavailable 716-day Mojito realization. It uses `tdis/A2` from the 30-day Ollie L1 file, Fourier brick-wall resampled by 20 from dt=0.25 s to dt=5 s. The 1-week case uses the first 120,960 samples and the 1-month case all 518,400 samples.

The estimator uses band 1e-4--0.1 Hz, seed 0, `centered=True`, 30 interior frequency knots, and the exact `1_week` and `1_month` settings in `mojito_experiments.EXPERIMENTS` (300 warmup + 300 posterior draws in each of two sequential chains). The VI arm uses the current defaults: 2,000 AutoDiagonalNormal SVI steps at learning rate 1e-2. Each arm runs in a fresh Python process so compiled JAX state is not shared.

The thresholds and exact case settings were frozen before results in `PROTOCOL.json`; completed arms additionally record their script, inference, and VI source hashes.

Total estimator time covers WDM transformation, basis/PLS setup, optional VI, NUTS, and surface reconstruction, but excludes HDF5 loading. Tree-depth saturation means `num_steps >= 2^10 - 1`. The 90% log-surface interval is converted to an approximate posterior standard deviation for paired-surface normalization.

## Exact launch commands

```bash
uv run python studies/ollie_tdi/benchmark_vi_warmstart.py --proxy --preflight
uv run python studies/ollie_tdi/benchmark_vi_warmstart.py --proxy
uv run python studies/ollie_tdi/benchmark_vi_warmstart.py --proxy --report-only
```

Completed outputs are preserved on rerun. Use `--force` only to deliberately replace an existing arm.

## Results

### 1_week

- Total: no-VI 19.9 s; VI 22.9 s (speedup -15.4%).
- NUTS: no-VI 18.2 s; VI 17.4 s; SVI itself 3.9 s.
- Divergences: 0 -> 0; tree saturation: 0.0% -> 0.0%.
- Mean accept probability: 0.890 -> 0.885; mean/max NUTS steps: 26.2/31 -> 21.5/31.
- Minimum ESS: 214.1 -> 121.6; maximum R-hat: 1.0193 -> 1.0429.
- Paired log-surface RMS: 0.00323; normalized RMS: 0.028; cells beyond one pooled SD: 0.00%.
- Phi-time mean (SD): 84.9 (9.47) -> 84.9 (9.89); phi-frequency: 4.52 (0.635) -> 4.53 (0.625).
- VI guide median vs its eventual MCMC log surface: RMS 0.01714, or 0.215 posterior SD pointwise-normalized RMS.
- Predeclared conclusion: `remove_vi_after_preserving_benchmark`.

### 1_month

- Total: no-VI 55.9 s; VI 63.1 s (speedup -12.9%).
- NUTS: no-VI 49.9 s; VI 49.6 s; SVI itself 7.3 s.
- Divergences: 0 -> 0; tree saturation: 0.0% -> 0.0%.
- Mean accept probability: 0.848 -> 0.884; mean/max NUTS steps: 15.0/15 -> 15.0/31.
- Minimum ESS: 164.6 -> 178.2; maximum R-hat: 1.0232 -> 1.0213.
- Paired log-surface RMS: 0.0009222; normalized RMS: 0.016; cells beyond one pooled SD: 0.00%.
- Phi-time mean (SD): 122 (10.1) -> 122 (10.1); phi-frequency: 2.44 (0.37) -> 2.44 (0.368).
- VI guide median vs its eventual MCMC log surface: RMS 0.03421, or 0.768 posterior SD pointwise-normalized RMS.
- Predeclared conclusion: `remove_vi_after_preserving_benchmark`.

## Overall conclusion

Do not enable VI; remove the VI path after preserving these artifacts: both cases show added cost without material diagnostic benefit.
