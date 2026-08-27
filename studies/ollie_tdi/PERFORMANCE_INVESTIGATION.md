# TVPsplinePSD performance investigation

## Executive conclusion

On the representative exact Ollie WDM grid, inference speed is controlled by
two different regimes:

1. With the wrong non-centered hierarchy, NUTS geometry dominates everything.
   The non-centered run was 11.4x slower than the centered run, saturated the
   tree depth in 50% of draws, and did not produce a valid posterior.
2. With the centered hierarchy, each trajectory is stable and the main cost is
   repeated value-and-gradient evaluation over the 647,520-cell likelihood
   grid. The previous left-associated tensor contraction did unnecessary work.
   Replacing it with an optimized `einsum` reduced matched two-chain NUTS time
   from 60.26 s to 36.61 s (1.65x) without changing trajectories or diagnostics.

The optimized centered model is now a mixture of elementwise/memory-bandwidth
and autodiff cost, multiplied by the NUTS trajectory length. It is not primarily
limited by the first small matrix multiplication or by the number of sampled
coefficients.

Eigenmode truncation is scientifically plausible at 75% retention in this one
pilot, but it saves only 3.5% wall time. More aggressive truncation does not
speed up inference and eventually damages the moving-null structure. Native
sparse B-spline evaluation is slower in JAX and gives much worse NUTS geometry.
Neither approximation should replace the default model based on this evidence.

## Benchmark problem and host

- Checkout: local `origin/main` at `0efb6d2d9ae487008334ad2e436ca6915407e583`
  before the changes in this investigation.
- Data: cached 30-day Ollie A2 WDM coefficients from
  `studies/results/ollie_tdi/knot_map_benchmark/wdm_coeffs.npz`.
- Likelihood grid: 120 time rows x 5,396 frequency channels = 647,520 cells.
- Basis: 20 time x 98 frequency functions = 1,960 spline coefficients.
- Sampled parameters: 1,962 (`s`, `log(phi_time)`, and `log(phi_freq)`).
- Backend: JAX 0.10.1, NumPyro 0.21.0, CPU float64.
- Host: Apple M4 Pro (`Mac16,8`), macOS arm64.
- Model: exact, unbinned WDM Whittle likelihood; centered tensor P-spline;
  linear knots; `target_accept_prob=0.85`; `max_tree_depth=10`.

The reproducible benchmark artifacts are:

- `studies/results/ollie_tdi/performance/hot_path.json`
- `studies/results/ollie_tdi/performance/eigenmode_truncation.json`
- `studies/results/ollie_tdi/performance/sparse_basis_prototype.json`

These result files are generated/ignored; the scripts that produce them are
version-controlled beside this report.

## Representative centered NUTS benchmark

The compilation-separated full-model run used two sequential chains, 100
warmup transitions per chain, and 150 retained draws per chain. An identical
first warmup was discarded to populate the JAX cache.

| Quantity | Result |
|---|---:|
| Estimated first-compilation overhead | 2.73 s |
| Cached warmup time | 8.80 s |
| Cached sampling time | 9.88 s |
| Cached NUTS total | 18.67 s |
| Mean / median `num_steps` | 31 / 31 |
| Max-tree-depth hits | 0% |
| Divergences | 0 |
| Median / minimum coefficient ESS | 404.5 / 21.8 |
| Max coefficient R-hat | 1.040 |
| `phi_time` ESS / R-hat | 154.5 / 0.997 |
| `phi_freq` ESS / R-hat | 651.5 / 0.995 |

Wall time per effective sample, using the cached NUTS total, was 0.046 s for
the median spline coefficient, 0.121 s for `phi_time`, and 0.029 s for
`phi_freq`. This is a short engineering run; the matched 200-warmup/300-draw
two-chain confirmation had max coefficient R-hat 1.016 and zero divergences.

## Hot-path profile

All times below are median cached JIT times at the production dimensions.

| Component | Forward | Value + gradient |
|---|---:|---:|
| `B_t @ W` only | 0.023 ms | -- |
| `W @ B_f.T` only | 0.119 ms | -- |
| Full optimized surface | 0.237 ms | -- |
| Complete Whittle elementwise/reduction on a surface | 0.366 ms | -- |
| Complete optimized log likelihood | 0.590 ms | 1.031 ms |
| Complete potential energy, including priors | 0.605 ms | 1.017 ms |

The Whittle subcomponents were 0.046 ms for `counts * log_psd`, 0.280 ms for
`power * exp(-log_psd)`, and 0.111 ms for the reduction when measured
separately. Fusion means these component timings should not be added to predict
the complete likelihood exactly.

The complete potential has essentially the same gradient time as the
likelihood, so prior bookkeeping is negligible. A 31-step trajectory predicts
about 31.5 ms per retained draw from the potential benchmark; the measured
sampling phase was 32.9 ms per draw. This agreement identifies repeated
value-and-gradient calls, rather than Python or MCMC bookkeeping, as the
centered sampler's hot path.

The current optimized regime is therefore:

- elementwise/exponential and memory traffic are slightly more expensive than
  the optimized surface contraction;
- reverse-mode autodiff roughly doubles the complete forward cost;
- NUTS trajectory length multiplies that cost by about 31;
- parameterization can still overwhelm all kernel improvements if it lengthens
  or invalidates the trajectories.

## Tensor contraction strategies

For `(N_t, K_t, K_f, N_f) = (120, 20, 98, 5396)`, explicit left association
requires about 63.7 million multiply-add terms, versus 23.5 million for right
association, a 2.71x operation-count ratio.

| Strategy | Surface | Complete likelihood | Value + gradient |
|---|---:|---:|---:|
| `(B_t @ W) @ B_f.T` | 0.365 ms | 0.871 ms | 1.730 ms |
| `B_t @ (W @ B_f.T)` | 0.232 ms | 0.612 ms | 1.245 ms |
| optimized `einsum("ti,ij,fj->tf")` | 0.231 ms | 0.610 ms | 1.074 ms |

The optimized `einsum` was consistently best, especially in the backward
pass. It is now used through `tensor_product_surface` in the core, nested,
joint, and separable multichannel likelihood paths.

The matched end-to-end confirmation used two chains with 200 warmup and 300
draws per chain, the same seed, and no statistical-setting changes:

| Metric | Old left contraction | Optimized `einsum` |
|---|---:|---:|
| NUTS runtime | 60.26 s | 36.61 s |
| Mean / median steps | 31 / 31 | 31 / 31 |
| Divergences / depth hits | 0 / 0% | 0 / 0% |
| Max coefficient R-hat | 1.016 | 1.016 |
| Median coefficient ESS/s | 19.32 | 31.38 |
| `phi_time` ESS/s | 8.01 | 11.90 |
| `phi_freq` ESS/s | 26.72 | 36.81 |

This is a 39.2% wall-time reduction and 1.62x median-coefficient ESS/sec gain.

## NUTS geometry

The centered/non-centered comparison used the optimized contraction, the same
problem, seed, two chains, 100 warmup transitions, and 100 retained draws.

| Metric | Centered | Non-centered |
|---|---:|---:|
| NUTS runtime | 23.35 s | 266.57 s |
| Mean / median steps | 63 / 63 | 551 / 879 |
| Max-depth-hit fraction | 0% | 50% |
| Divergences | 0 | 100 |
| Max coefficient R-hat | 1.040 | 22.38 |
| Median coefficient ESS/s | 13.19 | 0.0063 |
| `phi_time` ESS/s / R-hat | 5.77 / 0.998 | 0.035 / 1.44 |
| `phi_freq` ESS/s / R-hat | 6.96 / 1.003 | 0.0043 / 2.97 |

The non-centered result is invalid, not merely slow. Its likelihood-pinned
coefficients must all rescale when either smoothing precision moves, producing
the known hierarchical funnel. Centered sampling is the main production
recommendation for grids of this size.

A short centered pilot compared the default diagonal mass matrix with a dense
2x2 block for `(phi_time, phi_freq)`. The block increased trajectory length
from 47 to 63 and wall time from 26.1 s to 30.0 s. Coefficient ESS/sec improved
in that realization, but both smoothing-precision ESS/sec values decreased.
That mixed, seed-sensitive result does not support changing the default. A full
1,962-dimensional dense mass matrix is not sensible here: it discards the
already useful eigenbasis scaling and introduces quadratic storage and dense
factorization during adaptation.

No `target_accept_prob` or tree-depth reduction is recommended. The centered
model already has zero divergences and no depth saturation; lowering either
setting would trade posterior validity for apparent speed. The existing
penalized-least-squares initialization also reaches stable centered sampling.

## Roughness-eigenmode truncation

Modes were ranked by ascending prior-mean precision,
`E[phi_t] lambda_t + E[phi_f] lambda_f`, with `E[phi_t] = E[phi_f] = 2` under
the configured Gamma priors. This is a principled low-roughness ranking. The
omitted modes were fixed to zero, while the exact likelihood grid and retained
mode priors were unchanged. Each fit used two chains, 100 warmup transitions,
and 150 draws.

| Retained | Parameters | NUTS time | Speedup | Mean steps | Coeff. ESS/s | Normalized surface RMS | CI width ratio | Null-corridor log RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100% | 1,962 | 18.67 s | 1.00x | 31.0 | 21.66 | 0.000 | 1.000 | 0.000 |
| 75% | 1,472 | 18.04 s | 1.04x | 31.0 | 28.16 | 0.169 | 0.975 | 0.007 |
| 50% | 982 | 18.80 s | 0.99x | 30.7 | 29.53 | 0.396 | 0.863 | 0.026 |
| 25% | 492 | 20.26 s | 0.92x | 47.4 | 17.41 | 3.981 | 0.628 | 0.527 |

At 75%, the mean absolute log-PSD difference was 0.0067, the 99th-percentile
relative PSD difference was 4.8%, and 0.42% of cells shifted by more than one
full-model posterior SD. The temporal-gradient disagreement was 0.0028 log-PSD
units RMS. These are encouraging approximation diagnostics, but the 3.5%
runtime gain is too small to matter relative to Monte Carlo variability and the
new contraction gain.

At 50%, interval widths shrink by about 14% and surface disagreement grows even
though runtime does not improve. At 25%, 70.7% of cells move by more than one
baseline posterior SD, the 99th-percentile relative PSD difference reaches
145%, and the narrow moving-null corridors degrade much more than the rest of
the surface. The posterior is partially, but not usefully for wall time,
low-rank under this implementation.

The reason is empirical: evaluating the full 647,520-cell likelihood still
costs the same, and trajectory length does not fall at 75% or 50%. An irregular
roughness-ranked mode mask also breaks rectangular tensor separability; the
prototype scatters retained parameters into the full coefficient matrix to
preserve the fast tensor contraction. A future reduced model would need a
different structured low-rank basis to reduce both parameter and likelihood
work, followed by simulation-truth validation of sharp/time-varying features.

## Native sparse B-spline prototype

Cubic native B-spline rows were stored as four active indices and four values.
This reduced basis storage from 4.25 MB for the dense whitened bases to 0.265 MB
(16.0x). The evaluated spline surface was numerically identical after rotating
the same coefficients between native and eigen coordinates.

| Kernel | Dense whitened | Native sparse |
|---|---:|---:|
| Forward surface | 0.237 ms | 0.565 ms |
| Complete likelihood value + gradient | 1.050 ms | 1.815 ms |

On this CPU/XLA backend, gather/reduction traffic costs more than the dense
BLAS-style contractions. More importantly, sampling native coefficients exposes
the correlated structured roughness prior. A deliberately short two-chain
25-warmup/50-draw pilot had 50 divergences, max coefficient R-hat 3.73, and
median coefficient ESS 4.35. This pilot is too short for posterior conclusions,
but it is sufficient to reject the prototype as a performance direction: both
the kernel and geometry are worse.

## Recommendations

1. Keep the optimized `tensor_product_surface` contraction. It is exact,
   low-risk, tested, and gives the largest valid wall-clock gain found here.
2. Make `centered=True` the explicit choice in every production-like large-grid
   study. Consider changing the package default only with a compatibility plan,
   because weak/small-data problems can still prefer non-centered sampling.
3. Keep the diagonal mass matrix, `target_accept_prob=0.85`, depth 10, and the
   current PLS initialization unless a longer replicated study shows an
   ESS/sec improvement for a specific alternative.
4. Do not add eigenmode truncation to the default model. If approximation is
   desired, treat 75% retention as a candidate requiring multiple simulated
   realizations, production chains, truth-based coverage, and explicit moving-
   null/rapid-variation checks. Expect little wall-time benefit without a basis
   that also reduces surface-evaluation work.
5. Do not rewrite inference in native sparse coefficient space on this backend.
   The memory saving is real but small in absolute terms, while forward,
   gradient, and NUTS geometry all worsen.
6. After the contraction fix, the next exact-model optimization target is the
   fused Whittle exponential/reduction and its reverse pass, not the small first
   contraction. Backend-specific GPU tests may change the balance and should be
   measured independently.

## Evidence boundary

This is a single cached real-data realization on one Apple CPU backend. The
exact contraction result is strong because it is algebraically identical,
microbenchmarked forward/backward, and confirmed in matched two-chain NUTS runs.
The eigenmode and mass-matrix results are engineering pilots, not production
coverage studies. No approximate model default or scientific recovery claim is
supported without replicated simulation truth and longer chains.

Coarse-graining remains an established baseline from the separate repository
study, but it was deliberately not the focus here.
