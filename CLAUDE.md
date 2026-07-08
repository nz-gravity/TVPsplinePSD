# CLAUDE.md

Guidance for Claude (and humans) working in this repo. See `README.md` for the
scientific overview; this file captures working conventions, the active
Mojito non-stationarity study, and hard-won gotchas.

## What this is

`tv_pspline_psd` is a Bayesian estimator for **non-stationary** noise PSDs. It
fits a smooth log-PSD *surface* `log S(t,f) = B_t W B_f^T` to time-frequency
coefficients under a Whittle likelihood `c ~ N(0, S)` with an anisotropic
P-spline roughness prior. Front ends are representation-agnostic: `run_wdm_psd_mcmc`
(WDM wavelet, one real coeff/cell) and `run_stft_mcmc` (STFT, real+imag) share
the same core surface fitter (`fit_log_pspline_surface` in `tv_pspline_psd/inference.py`).

Core entry points: `run_wdm_psd_mcmc`, `PSplineConfig`, `wdm_analysis_coefficients`,
`summarize_mcmc_diagnostics`, `set_paper_style` (all exported from `tv_pspline_psd`);
`wdm_white_noise_calibration` (from `datasets`). Sampler is NumPyro NUTS; the fit
returns posterior `psd_mean/lower/upper` (coefficient-variance units) plus the raw
`coeffs`, `power`, `time_grid`, `freq_grid`.

## Estimator conventions & gotchas

- **WDM sizing**: `TimeSeries(x, dt).to_wdm(nt=...)` requires **`nt` and `nf = N/nt`
  both even**, and `N` divisible by `nt`. Crop the series to the largest valid
  `nt*nf` before transforming. Nyquist = `1/(2 dt)`; channel spacing `df = 1/(2 nf dt)`.
- **`centered=True` on large grids.** The default non-centered parameterization
  freezes `phi` (saturated tree depth, n_eff~1) once the likelihood pins the
  coefficients — i.e. on any grid with many cells (≳10⁵). Set `PSplineConfig(centered=True)`
  for real data / long segments. Small/weak-data problems can stay non-centered.
- **Scale-free power floors.** Never add an absolute floor like `log(power + 1e-30)`;
  TDI amplitudes are ~1e-20 so `coeffs²` ~ 1e-34 and a fixed floor flattens the whole
  surface. Guard `log` with a value relative to the data (e.g. smallest positive power).
- **Frequency-band restriction** maps to channel trims: `trim_low = ceil(fmin/df)`,
  `trim_high = nf - floor(fmax/df)` (see `band_trims` in `studies/ollie_tdi/fit_mojito_segment.py`).
  With `fmax` at Nyquist, `trim_high = 0`.
- **`np.einsum` must use `optimize=True`** for 3+ operands on big grids. Without it
  numpy runs a naive element-wise kernel that scales catastrophically (a 2-year
  surface reconstruction hung for hours; `optimize=True` = ~2000x faster). Fixed in
  `surface_summaries`; watch for the same pattern elsewhere.

## Active study: Mojito non-stationary TDI noise (`studies/ollie_tdi/`)

Testing how LISA's drifting TDI transfer-function null (from the flexing
constellation) folds into WDM-domain noise estimation, en route to a **PRD paper**.

**Data (external, ~6.8 GB):** `../MojitoProcessor/Mojito_Data/processed_segments_noise_no_segmentation.h5`
(sibling repo — add `mojito-processor` via `uv add mojito-processor`, note the hyphen).
`processed/segment0/{X,Y,Z}` = 2nd-gen TDI, **dt=5 s (fs=0.2 Hz), N=12,370,796,
~716 days ≈ 1.96 yr**, one continuous segment, Tukey-tapered (α=0.05 → first/last
~18 days ramp from zero). Time-varying arm light-travel-times in
`raw/segment0/ltts/{ij}` (arms flex 1.4–1.8% peak-to-peak).

**Physics established:** the first Michelson null sits at `f = 1/(2⟨L⟩) ≈ 0.06 Hz`
and drifts ~1.4% annually (≈100 WDM bins), but that's sub-pixel on a log axis. The
null is set by the mean one-way time of the **four arms adjacent to the channel's
spacecraft** (X↔S/C1: 12,21,13,31), not one arm. A 2nd null at ~0.03 Hz comes from
the 2nd-gen TDI factor. Fourier/Whittle PSD estimation biases for segments **>~1 day**
(a moving spectral feature gets smeared across resolution bins; `T* ≈ ḟ_null^(-1/2) ≈ 1 day`);
WDM is robust because each cell only needs *local* stationarity.

**Scripts (in `studies/ollie_tdi/`):**
- `check_XYZ.py` — raw WDM log-power spectrograms (X/Y/Z + orthogonal A/E/T), `--days`/`--start-day`; taper-aware windowing.
- `null_zoom.py` — linear-frequency zoom on the first null with the `1/(2⟨L⟩)` overlay (per-channel arm map in `SC_ARMS`).
- `fit_mojito_segment.py` — windowed TV-PSD MCMC fit; `--start-day`/`--days`/`--end-day`, band `--fmin`/`--fmax`, `--channel`.
- `mojito_experiments.py` — the 5-length ladder (`1_week`,`1_month`,`6_month`,`1_year`,`1.5_year`,`full`); each writes `spectrum/null_zoom/whitening/surface.png`, `fit.npz`, `diag.json` to `studies/results/ollie_tdi/experiments/<name>/`. **The whitening panel (z=w/√Ŝ hist + z̄² per time/freq) is the core goodness-of-fit check.**
- `mojito_validation.py` — cross-validated out-of-sample whitening (#8) + AET cross-channel correlation (#5).
- `aet_corr_summary.py` — AET `corr(f)` overlaid across all window lengths (coefficient-level; Pearson corr of WDM coeffs is scale-invariant so no MCMC needed, and it matches the MCMC version).
- `fit_ollie_tdi.py` — pre-existing reference fit on `datasets/ollie_data`.

**Findings:** whitening passes at all lengths (mean z²≈1.00, 0 divergences). Out-of-sample
cross-val passes (PSD generalizes / stationary at short scales). **AET essentially
uncorrelated at every length** (|overall r|≤0.008) → diagonal covariance OK for PE.
Mild positive excess kurtosis (~0.25–0.31) in residuals, likely null-region misfit —
watch for PE false alarms. `full` cross-val must be a **year-1-fit → year-2-whiten**
split (no room for a disjoint 716-day window).

**Fit benchmarks** (band 1e-4–0.1 Hz, `centered=True`, `n_interior_knots_freq=30`): grid
cells ≈ `nt*nf ≈ N`; cost ~linear-to-slightly-superlinear. 1wk 61s, 1mo 261s (2 chains),
6mo 630s, 1yr 1414s, full 2942s (single chain, ~46 min NUTS + fast post-processing).

## Operational notes

- **Background jobs**: launch long MCMC with the harness's `run_in_background` **only**
  — do NOT also wrap in `nohup ... &` (double-backgrounding orphans the worker onto one
  core and it gets killed on teardown). Don't pipe the launch through
  `grep -avE "it/s"` if you want a live progress bar — that filters the bar out of the
  captured output.
- **Tracking a running fit**: the NumPyro bar uses carriage returns, so
  `tail -c 4000 <task-output> | tr '\r' '\n' | grep -E "warmup|sample" | tail -1`.
  `[elapsed<remaining]` in the bar is the live ETA (NUTS only; add ~1–2 min for
  calibration + plots). Nothing lands in the output dir until the run fully finishes.
- **Calibration/plots** run silently after the sampler; `full/`-size `pcolormesh` over
  ~12M cells is slow — consider rasterizing/coarsening the surface plot for 2-year runs.
- Prefer `set_paper_style()` and match `fit_ollie_tdi.py`'s figure idioms.
- Tests: `python -m pytest tests/ -q` (note: `test_io.py` needs `netCDF4`/`h5netcdf`,
  which may be absent — unrelated to core fits).

## Direction (paper)

Targeting PRD. Plan discussed: demonstrate downstream **PE impact** with an **EMRI**
(not a monochromatic GB — an EMRI's time-frequency track sweeps the band and crosses
the drifting null), via localized PE (start near truth to sidestep the multimodal
search) cross-checked against a Fisher/systematic-bias estimate; then joint
signal+noise refinement (`joint.py`), with the smoothness prior separating the sharp
coherent signal from the smooth noise drift. FEW (EMRI waveforms) is CuPy-based and not
JAX-differentiable — treat it as an external template oracle (CPU FEW + finite-diff/Fisher);
keep it out of the differentiable JAX noise path.
