# VI warm-start benchmark and removal decision

## Decision

The optional mean-field variational-inference (VI) warm start was removed. In
paired 1-week and 1-month proxy runs it increased total estimator time by
13--15%, did not materially improve NUTS diagnostics, and left the converged
MCMC posterior unchanged. Both cases met the removal criterion declared before
the benchmark.

This file is the durable study record. The larger local JSON/NPZ evidence is
retained under `studies/results/ollie_tdi/vi_benchmark/`; it is not required by
the package at runtime.

## Data and preprocessing

The intended 716-day Mojito realization was unavailable. Its original blocked
protocol and report remain at:

- `studies/results/ollie_tdi/vi_benchmark/PROTOCOL.json`
- `studies/results/ollie_tdi/vi_benchmark/REPORT.md`

The executed benchmark used an explicitly authorized and labelled
`ollie_30day_proxy`, not Mojito data:

- File: `datasets/ollie_data/simulated_noise_30_days_L1_ext.h5`
- Dataset: `tdis/A2`, 10,368,000 samples at `dt=0.25 s`
- Dataset SHA-256:
  `3b2d9ee21678cce9dacac12f3b64557b3752a5856f8490d04a572436f3a803e3`
- Preprocessing: `scipy.signal.resample`, Fourier-domain brick-wall resampling
  by 20 to `dt=5 s`
- Result: 518,400 samples; SHA-256:
  `e287b38ab82739c116b42bd60ff92c9e6486274094902da2e9a98b33f92ecf9e`
- The 1-week case used samples `[0:120960]`; the 1-month case used all
  518,400 samples. These match the Mojito study's sample counts.

## Method

Each arm ran in a fresh Python process with the same data, seed, and existing
experiment configuration. Both cases used `centered=True`, channel-band
equivalent frequencies `1e-4--0.1 Hz`, 30 interior frequency knots, two
sequential chains, 300 warmup draws, 300 posterior draws, seed 0,
`max_tree_depth=10`, and target acceptance 0.85. Time settings came directly
from `mojito_experiments.EXPERIMENTS`: `nt=32` and 8 time knots for 1 week;
`nt=30` and 10 time knots for 1 month.

The VI arm added the then-current `AutoDiagonalNormal` guide with 2,000 SVI
steps at learning rate `1e-2`; its median initialized NUTS. VI draws were never
used as MCMC posterior draws. Total time includes WDM transformation, basis and
penalized-least-squares setup, optional VI, NUTS, and surface reconstruction,
but excludes HDF5 loading.

The predeclared removal rule required posterior equivalence, at least 5% extra
total cost, and no material sampling improvement. Posterior equivalence required
pointwise-normalized log-surface RMS at most 0.25, at most 1% of cells beyond
one pooled posterior standard deviation, and each phi mean shift at most 0.5
pooled posterior standard deviations.

## Results

| Metric | 1 week, no VI | 1 week, VI | 1 month, no VI | 1 month, VI |
|---|---:|---:|---:|---:|
| Total estimator time (s) | 19.86 | 22.92 | 55.88 | 63.11 |
| NUTS time (s) | 18.17 | 17.40 | 49.93 | 49.64 |
| VI time (s) | 0 | 3.87 | 0 | 7.27 |
| Divergences | 0 | 0 | 0 | 0 |
| Tree-depth saturation | 0% | 0% | 0% | 0% |
| Mean accept probability | 0.890 | 0.885 | 0.848 | 0.884 |
| Mean / maximum NUTS steps | 26.2 / 31 | 21.5 / 31 | 15.0 / 15 | 15.0 / 31 |
| Minimum ESS across sites | 214.1 | 121.6 | 164.6 | 178.2 |
| Maximum R-hat across sites | 1.0193 | 1.0429 | 1.0232 | 1.0213 |
| Phi-time mean (SD) | 84.9 (9.47) | 84.9 (9.89) | 121.9 (10.11) | 121.6 (10.08) |
| Phi-frequency mean (SD) | 4.521 (0.635) | 4.526 (0.625) | 2.441 (0.370) | 2.442 (0.368) |

VI increased total time by 15.4% for 1 week and 12.9% for 1 month. The paired
MCMC log-surface differences were much smaller than posterior uncertainty:

| Posterior comparison | 1 week | 1 month |
|---|---:|---:|
| RMS log-surface difference | 0.003230 | 0.000922 |
| Pointwise-normalized RMS | 0.0279 | 0.0159 |
| Cells beyond one pooled SD | 0% | 0% |
| Maximum phi mean shift, pooled SD | 0.0051 | 0.0171 |
| VI guide vs eventual MCMC, normalized RMS | 0.215 | 0.768 |

Thus the VI point estimate was adequate as an initializer but did not reduce
NUTS cost enough to repay its own optimization. The analytic penalized-
least-squares initialization already produced well-behaved sampling.

## Provenance and retained evidence

The four arms were run from Git commit
`bd384bd21a85c4ede2f2a25c77f9b1cae507fd80` with an intentionally dirty study
worktree. Source hashes recorded by every arm were:

- Benchmark runner SHA-256:
  `3ffc304fe1dc4c42adec460074e1884bbfd8785bc43d6d428a4427a8b748d222`
- `tv_pspline_psd/inference.py` SHA-256:
  `b26a6b1b026041ba3c2be42564de440515f52d91cbf62357ae2577459f2dd9f1`
- Removed `tv_pspline_psd/vi.py` SHA-256:
  `c87abdcae8feb409f522fcd8bd3612289f3555aef9d2b0d8e7a8ccdd45f919c3`

Detailed evidence is retained locally in
`studies/results/ollie_tdi/vi_benchmark/ollie_30day_proxy/`:

- `PROTOCOL.json` and `REPORT.md`
- `input_manifest.json` and the hashed decimated input array
- Per-arm `metrics.json` and `arrays.npz`
- Per-case `comparison.json`

These artifacts contain the complete configurations, package versions, git
state, timing breakdowns, extra-field draws, phi samples, VI losses and guide
surface, and paired posterior summaries needed to audit or reproduce the
decision.
