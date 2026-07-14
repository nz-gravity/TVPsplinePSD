# Adaptive knot allocation study

Status: the original allocator was committed to `main` in `0cd9357`. The
frequency-strategy configuration policy below is the current working-tree
change and still needs its own commit.

## What was added

- `tv_pspline_psd/adaptive_knots.py`: a Whittle-MAP pilot allocator using an
  analytic L-BFGS-B gradient, exact power/count coarsening, held-out deviance,
  curvature/deviance/hybrid density quantiles, and minimum knot spacing.
- `fit_log_pspline_surface` now accepts explicit interior time/frequency knots
  and validates/persists their physical grid coordinates.
- Benchmark, centered-NUTS confirmation, and known-truth coverage scripts are
  under `studies/ollie_tdi/` and `studies/knot_coverage_simulation.py`.

## LISA 30-day result

On the A2 30-day Ollie segment, adaptive explicit frequency knots improved
held-out performance at the same basis size:

- centered NUTS overall deviance: `0.589679 -> 0.563735` (4.4% lower);
- null-region deviance: `0.490217 -> 0.290311` (40.8% lower);
- frequency z² RMSE: `0.366210 -> 0.295826` (19.2% lower);
- residual excess kurtosis: `0.16144 -> 0.03580` (~78% lower);
- zero divergences in both fits; adaptive sampling took about 8% longer.

In eight independent drifting-notch simulations, notch coverage improved in
all replicates (`0.57572 -> 0.67751` on average) and notch log-MSE decreased
(`0.30259 -> 0.21666`).

The selected knots are fixed before NUTS (empirical-Bayes pilot); direct free
knot-coordinate gradient optimization was not pursued because the quantile
allocator captures the gain while avoiding collision/local-mode issues.

## Public selection policy

Time knots are always linear unless a caller supplies explicit physical time
knots for a controlled experiment. Frequency placement is selected with
`PSplineConfig(freq_knot_strategy=...)`:

- `"adaptive"` (the default): curvature knots from the Whittle-MAP pilot;
- `"linear"`: uniform spacing in Hz, for a baseline or exact legacy control;
- `"log"`: uniform spacing in log Hz, requiring a strictly positive analysis
  frequency grid.

Explicit `interior_knots_freq` always takes priority over the configured
strategy. The selected strategy and final physical knots are retained in fit
results and provenance.

## Verification and cleanup

The full suite passed before cleanup (`44 passed`). Per request, the two
dedicated experimental test files were removed; the remaining suite passes
after cleanup. After introducing the strategy policy, it passes with
`33 passed in 15.01s`. Production implementation and benchmark artifacts
remain.
