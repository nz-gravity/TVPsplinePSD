# tv_pspline_psd

Bayesian estimation of **non-stationary** noise power spectral densities from
Wilson–Daubechies–Meyer (WDM) wavelet coefficients, using a tensor-product
**log-P-spline** surface.

The estimator fits a smooth log-PSD surface

```
log S(t, f) = B_t W B_f^T
```

to the time-frequency coefficients with a Gaussian Whittle likelihood
(`c ~ N(0, S)`) and an anisotropic P-spline roughness prior. The estimator is
representation-agnostic: `run_wdm_psd_mcmc` feeds it one real coefficient per
cell (WDM), `run_stft_mcmc` feeds it the real and imaginary parts of a
short-time Fourier coefficient (the moving periodogram) — same prior, same
likelihood, different front end (`tv_pspline_psd.fit_log_pspline_surface`). The prior is
sampled in a **non-centered (whitened)** form — standard-normal coefficients in
the penalty eigenbasis, rescaled by `1/sqrt(phi_t λ_t + phi_f λ_f)` — which
removes the `phi`-vs-weights funnel that otherwise causes vanishing gradients and
maximal leapfrog/tree-depth in NUTS. The smoothing precisions are sampled on the
log scale with a `Gamma` hyperprior.

## Layout

| Path | Role |
|------|------|
| `tv_pspline_psd/` | the estimator package (splines, prior/model, inference, metrics, plotting) |
| `datasets/` | data generators, decoupled from the estimator (`ls2`, `lisa`) |
| `studies/` | simulation studies (LS2 ×100, LISA confusion noise) |
| `examples/` | minimal quickstart |
| `notes/` | manuscript (`ms.tex`), figure scripts (`scripts/`), figures (`figures/`) |

Data generation is intentionally separate from estimation: each dataset module
returns a raw time series (and an analytic true PSD) and knows nothing about the
spline machinery.

## Install

```bash
uv sync                  # installs deps and the tv_pspline_psd + datasets packages
# or: uv pip install -e .
```

The package must be installed before running the scripts below: they import
`tv_pspline_psd` and `datasets` directly (no `sys.path` manipulation).

## Quickstart

```python
import numpy as np
from tv_pspline_psd import PSplineConfig, run_wdm_psd_mcmc
from datasets import simulate_ls2

data = simulate_ls2(576, rng=np.random.default_rng(0))
results = run_wdm_psd_mcmc(data, dt=0.1, nt=24, config=PSplineConfig())
psd_surface = results["psd_mean"]        # (n_time, n_freq) posterior-mean PSD
```

Or run `python examples/quickstart.py`.

## Simulation studies

```bash
python studies/ls2_simulation_study.py            # 100 LS2 repeats (~5 min)
python studies/ls2_simulation_study.py --quick    # fast smoke run

python studies/lisa_study.py                       # single rich LISA fit + figures
python studies/lisa_study.py --repeats 30          # LISA error distribution
python studies/lisa_study.py --quick
```

Each study reports the log-PSD error against the Monte Carlo `E[w^2]` target
(what the estimator infers) and against the analytic PSD converted to
WDM-coefficient units via a per-channel white-noise calibration
(`E[w^2] = C_m · S_dig`). Results and figures are written to `studies/results/`.

## Manuscript figures

The figures used in `notes/ms.tex` are regenerated into `notes/figures/` by:

```bash
# Simulation study (Section 4)
python notes/scripts/make_sim_study_figures.py        # sim_three_panel, sim_mse_coverage

# LISA demonstration (Section 5): one annual realisation -> non-stationary figures
# (surface_comparison, gibbs_psd_bias, representation_comparison)
python notes/scripts/make_lisa_demo.py                # full one-year cyclostationary run
python notes/scripts/make_lisa_demo.py --quick        # fast smoke configuration
python notes/scripts/make_lisa_demo.py --render-only  # re-render figures from cached npz

# A/E multichannel ensemble (Section 5): unbiased GB amplitude + PSD coverage
python notes/scripts/make_lisa_ensemble.py            # 12 realisations -> lisa_ensemble
python notes/scripts/make_lisa_ensemble.py --render-only

# Time-localized source (Section 5): freq-domain Whittle vs WDM time-varying PSD
python notes/scripts/make_lisa_transient.py           # -> lisa_transient_corner

# Stationary-noise consistency corner (freq-domain vs WDM GB posteriors, A/E)
python studies/lisa_gb/make_corner.py --production    # lisa_gb_stationary_corner
```

## Joint signal + noise (global fit)

The Gaussian coefficient likelihood admits a signal mean, `c ~ N(h(θ), S)`, so a
signal and the non-stationary noise PSD can be inferred jointly. Two samplers are
provided: `run_joint_signal_noise_mcmc` samples both with a single NUTS
trajectory, while `run_gibbs_signal_noise_mcmc` runs a blocked **Metropolis-
within-Gibbs** scheme — a NUTS update of the noise PSD alternating with a NUTS
update of the signal amplitudes, each block adapting independently. A noise-only
or **stationary** fit (`run_stationary_psd_mcmc`, a time-invariant 1D
log-P-spline) instead absorbs the signal and the time-varying confusion power and
biases the PSD high. Demo:

```bash
python notes/scripts/make_lisa_demo.py   # jaxGB TDI signal + lisatools noise, blocked-Gibbs vs stationary
```

The realistic demo needs the `[lisa]` extra (`uv pip install -e '.[lisa]'`:
jaxGB + lisatools + lisaorbits). It generates the galactic-binary TDI response
(jaxGB) and the instrument-plus-modulated-confusion noise (lisatools) in the
**same TDI channel, generation, and units** — `datasets/lisa_tdi.py` validates
that the time-domain SNR matches the analytic value and that the noise
realizations reproduce the lisatools PSD, so the signal response and noise
generation are physically consistent.

## The LISA dataset

`datasets/lisa.py` builds non-stationary noise as

```
x(t) = x_inst(t) + m(u) x_gal(t),   S(t, f) = S_inst(f) + m(u)^2 S_gal(f),
```

with analytic Robson–Cornish–Liu (2019) instrument and Galactic-confusion PSDs
and a seasonal envelope `m(u)` (`<m^2> = 1`) that modulates the confusion
foreground in time. This is the time-variation the WDM estimator recovers.
