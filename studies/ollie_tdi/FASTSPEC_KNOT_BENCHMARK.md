# FastSpec-style frequency-knot benchmark

## Result

A fixed-budget running-median linear-window allocator improves the 30-knot
frequency basis on the local 30-day Ollie A2 data.  It should move forward as
an opt-in production strategy, with a time-block aggregation pass added before
testing the full Mojito mission ladder.

The comparison uses the `1e-4--0.1 Hz` Mojito band, 16 time knots, 30 frequency
knots, identical derivative roughness penalties, and blocked held-out WDM time
rows.  Allocation and fitting see training rows only.

## Five-fold MAP screen

Values are mean +/- sample standard deviation across holdout offsets
`0, 2, 4, 6, 8`.  Lower is better.

| placement | null excess deviance | frequency z2 RMSE | excess kurtosis |
|---|---:|---:|---:|
| uniform | 0.7962 +/- 0.0055 | 0.4407 +/- 0.0061 | 0.3150 +/- 0.0257 |
| production curvature | 0.5851 +/- 0.0178 | 0.3816 +/- 0.0028 | 0.1719 +/- 0.0234 |
| running-median chi-square | **0.4287 +/- 0.0163** | **0.3325 +/- 0.0077** | **0.0830 +/- 0.0349** |

The chi-square allocator wins on all three metrics in every fold.  Relative to
production curvature, mean null-region deviance improves by 26.7% and frequency
whitening RMSE improves by 12.9%.

## Centered-NUTS confirmation

The confirmatory run uses two chains, each with 250 warmup and 250 retained
draws.  Both fits have zero divergences and `R-hat` approximately one.

| placement | null excess deviance | frequency z2 RMSE | excess kurtosis |
|---|---:|---:|---:|
| production curvature | 0.5892 | 0.3756 | 0.1773 |
| running-median chi-square | **0.4423** | **0.3299** | **0.0959** |

## Method

`fit_running_median_chi2_knots` collapses training power with a pointwise time
median, applies a scale-free frequency running median, and grows successive
frequency windows until a linear-fit chi-square crosses a threshold.  Threshold
bisection preserves the configured knot count, so the method changes placement
without changing NumPyro model dimension.  Existing minimum-spacing projection
guards against nearly coincident knots.

The default pilot window in this benchmark is `0.5 mHz`.  Results are stable
from `0.25--1 mHz`; a `2 mHz` window weakens the null improvement because it
begins smoothing away the feature.

## Reproduction

```bash
.venv/bin/python studies/ollie_tdi/knot_map_benchmark.py \
  --time-knots 16 --freq-knots 30 --fmax 0.1 --maxiter 1000 \
  --holdout-offset 4 \
  --strategies uniform module_freq chi2_freq \
  --output studies/results/ollie_tdi/knot_map_benchmark/fastspec_chi2_30_fold4.json

.venv/bin/python studies/ollie_tdi/knot_nuts_confirmation.py \
  --time-knots 16 --freq-knots 30 --pilot-freq-knots 30 \
  --pilot-penalty 0.05 --fmax 0.1 \
  --strategies adaptive_map chi2_freq \
  --n-warmup 250 --n-samples 250 --num-chains 2 \
  --outdir studies/results/ollie_tdi/knot_map_benchmark/nuts_fastspec_chi2_30
```

## Limitation and next step

The 30-day test can safely collapse over time because null drift is small over
that interval.  A full-mission allocator should run the same frequency pass in
several time blocks and aggregate the window endpoints; a single two-year
median spectrum could smear the annual drift.  Keep time knots uniform, then
compare pure chi-square and curvature/chi-square hybrid placement across the
Mojito length ladder before changing the default.
