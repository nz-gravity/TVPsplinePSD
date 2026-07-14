# Adaptive coarse-graining benchmark

## Conclusion

Adaptive frequency coarse-graining is effective on the real 30-day A2 WDM
grid. A conservative pilot rule reduced the likelihood grid by 7.6--11.5x and
the measured NUTS runtime by 3.4--4.2x while keeping the posterior-mean surface
within 0.13--0.20 pooled posterior standard deviations RMS of the exact fit.
Adding two-row time aggregation increased the total cell reduction to 15--23x
with little additional surface shift.

Uniform 12-channel frequency bins achieved a similar runtime but biased the
null corridors more strongly. Aggressive adaptive bins (20x frequency
reduction) crossed the useful accuracy boundary: 6.1% of fine-grid cells moved
by more than one pooled posterior standard deviation.

## Protocol

- Data: cached 30-day Ollie A2 WDM coefficients from
  `knot_map_benchmark/wdm_coeffs.npz`.
- Fine grid: 120 time rows x 5,396 frequency channels.
- Model: centered tensor P-spline, 16 time and 94 frequency interior knots.
- Sampler: one NUTS chain, 100 warm-up and 100 posterior draws, common seed.
- Pilot: Gaussian-smoothed fine-grid log power, used only to select shared
  frequency-bin boundaries; NUTS receives sums of the unsmoothed powers.
- Adaptive criterion: extend a frequency bin while the pilot log-PSD range at
  every time remains below a threshold, subject to a 64-channel cap.
- Validation: posterior comparisons and whitening are evaluated on the full,
  unbinned grid.

## Results

| Likelihood | Cell reduction | NUTS time | Speedup | Normalized surface RMS | Cells >1 pooled SD | Null-region log RMS |
|---|---:|---:|---:|---:|---:|---:|
| Exact | 1.0x | 19.07 s | 1.00x | 0.000 | 0.000% | 0.0000 |
| Uniform frequency x12 | 12.0x | 4.12 s | 4.63x | 0.378 | 3.45% | 0.0624 |
| Adaptive, range 0.2 | 7.6x | 5.57 s | 3.43x | 0.134 | 0.035% | 0.0102 |
| Adaptive, range 0.3 | 11.5x | 4.59 s | 4.16x | 0.196 | 0.405% | 0.0165 |
| Adaptive 0.3 + time x2 | 23.0x | 3.92 s | 4.87x | 0.214 | 0.372% | 0.0173 |
| Adaptive, range 0.5 | 20.4x | 4.33 s | 4.41x | 0.529 | 6.13% | 0.0382 |
| Adaptive 0.5 + time x2 | 40.7x | 2.85 s | 6.70x | 0.537 | 6.25% | 0.0386 |

All production-length benchmark fits had zero divergences and used a mean of
31 leapfrog steps. The runtime reduction therefore comes from cheaper gradient
evaluations, rather than a change in sampled geometry.

At the range-0.3 setting, adaptive bins averaged 6.0 channels inside the
+/-2 mHz null corridors and 17.0 channels elsewhere. This is the main reason
they outperform uniform 12-channel bins at almost the same total grid size.

Fine-grid whitening changed little under the conservative adaptive fits. For
the exact, range-0.3 adaptive, and range-0.3-plus-time fits, respectively:

- pooled mean z-squared: 0.9980, 0.9966, 0.9965;
- null-corridor mean z-squared: 1.0580, 1.0512, 1.0509;
- frequency-resolved z-squared RMSE: 0.2537, 0.2519, 0.2519.

## Interpretation

The range-0.2 setting is the conservative default from this experiment. The
range-0.3 setting offers a better speed/accuracy trade if a normalized surface
RMS around 0.2 is acceptable. Range 0.5 is too aggressive for posterior-quality
work despite its acceptable pooled whitening statistics; pooled whitening alone
does not reveal the local surface displacement.

Two-row time aggregation appears safe on this 30-day grid, but contributes a
smaller speed gain than frequency aggregation. Larger time factors should not
be inferred from this result and need a dedicated multi-year test against the
annual null motion.

This is a single-realization, short-chain engineering benchmark. Before using
coarse-graining for manuscript coverage claims, repeat with at least two
production chains, simulation truth, and the null-track/out-of-sample checks.

## Reproduction

```bash
uv run python studies/ollie_tdi/coarse_graining_benchmark.py \
  --warmup 100 --samples 100
```

The range-0.2 and range-0.5 sensitivity runs use `--adaptive-log-range 0.2`
and `0.5`. JSON artifacts are written under
`studies/results/ollie_tdi/coarse_graining_benchmark*.json`.
