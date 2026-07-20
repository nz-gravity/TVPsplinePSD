# tv_pspline_psd

Bayesian estimation of non-stationary noise power spectral densities from
Wilson-Daubechies-Meyer wavelet coefficients using a tensor-product
log-P-spline surface.

<p>
	<a href="https://github.com/nz-gravity/TVPsplinePSD/actions/workflows/python-tests.yml"><img alt="Python tests" src="https://img.shields.io/github/actions/workflow/status/nz-gravity/TVPsplinePSD/python-tests.yml?branch=main&label=tests"></a>
	<a href="https://github.com/nz-gravity/TVPsplinePSD/actions/workflows/latex-pdf.yml"><img alt="Manuscript build" src="https://img.shields.io/github/actions/workflow/status/nz-gravity/TVPsplinePSD/latex-pdf.yml?branch=main&label=manuscript"></a>
	<a href="https://raw.githubusercontent.com/nz-gravity/TVPsplinePSD/manuscript-pdf/ms.pdf"><img alt="Download manuscript PDF" src="https://img.shields.io/badge/download-manuscript%20pdf-0f766e?style=for-the-badge"></a>
</p>

The estimator fits a smooth log-PSD surface

```text
log S(t, f) = B_t W B_f^T
```

to time-frequency coefficients with a Gaussian Whittle likelihood,

```text
c ~ N(0, S)
```

and an anisotropic P-spline roughness prior. The implementation is
representation-agnostic: `run_wdm_psd_mcmc` uses one real coefficient per cell
from the WDM transform, while `run_stft_mcmc` uses the real and imaginary parts
of a short-time Fourier coefficient via the moving periodogram. Both front ends
share the same core surface fitter.

The prior supports both non-centered standard-normal coefficients rescaled by
`1 / sqrt(phi_t lambda_t + phi_f lambda_f)` and centered eigen-coefficients.
Small or weak-data problems generally suit the non-centered form; large grids
where the likelihood pins the surface generally require `centered=True`. The
smoothing precisions are sampled on the log scale with a `Gamma` hyperprior.

## Install

```bash
uv sync
```

For development and tests:

```bash
uv sync --group dev
```

The package must be installed before running the examples or studies because
they import `tv_pspline_psd` directly.

Optional LISA extras are available with:

```bash
uv pip install -e .[lisa]
```

The Mojito study dependency is isolated in the studies extra:

```bash
uv sync --extra studies
```

## Quickstart

```python
import numpy as np
from tv_pspline_psd.datasets import simulate_ls2
from tv_pspline_psd import PSplineConfig, run_wdm_psd_mcmc

data = simulate_ls2(576, rng=np.random.default_rng(0))
results = run_wdm_psd_mcmc(data, dt=0.1, nt=24, config=PSplineConfig())
psd_surface = results["psd_geometric_mean"]
```

Surface point estimates are posterior geometric means,
`exp(E[log S])`. The legacy `psd_mean` key remains as a deprecated alias for
one release.

For a fuller example with diagnostics and a saved figure:

```bash
uv run python examples/quickstart.py
```

## Repository Layout

| Path | Role |
|------|------|
| `tv_pspline_psd/` | Estimator package: splines, priors, inference, diagnostics, metrics, plotting |
| `tv_pspline_psd/datasets/` | Synthetic datasets and references |
| `examples/` | Minimal runnable examples |
| `studies/` | Reproducible LS2 and LISA study workflows; see [`studies/README.md`](studies/README.md) |
| `notes/` | Design and validation notes |

## Generated artifacts

The repository contains the code, Slurm launchers, and compact paper inputs
needed to reproduce the studies.  Datasets and generated sampler outputs,
plots, logs, and campaign summaries belong outside Git under
`studies/results/`.  The one exception is a deliberately retained small table
under `studies/paper_figures/figures/` when the renderer consumes it directly.

Data generation is intentionally separate from estimation: dataset modules
return raw time series and analytic or Monte Carlo references without depending
on the spline machinery.
