# Stellar-origin BBH + time-varying PSD: demonstration plan

**Goal.** Show that estimating a chirping stellar-origin black-hole binary (SOBH)
in the LISA band against a **time-varying** noise PSD (the WDM tensor-product
log-P-spline of this package) gives calibrated source posteriors, whereas the
standard **stationary** PSD model biases them — because the source sweeps a
diagonal track through the time-frequency plane and is measured against the
*local* (cyclostationary Galactic-confusion) noise, not the year-average.

This is the realistic, differentiable upgrade of the toy
`notes/scripts/make_lisa_transient.py`.

## Decisions (locked)

- **Waveform:** `ripple` (PyPI `ripplegw`), `IMRPhenomD` — jax-native and
  differentiable, so the *entire* pipeline (waveform + P-spline noise + Whittle
  likelihood) is one jax graph and the source block samples with **NUTS**, not a
  gradient-free fallback. Runs on CPU for smoke and CUDA (Colab / OzStar) with
  the `jax_enable_x64` config the package already sets.
- **LISA response:** analytic **long-wavelength approximation** — time-dependent
  antenna patterns `F+(t), Fx(t)` from the LISA orbit, plus the Doppler/phase
  modulation. ~20 lines of jax, fully differentiable; its annual modulation
  reinforces the non-stationarity story. (Full TDI is deferred — not needed for
  the argument.)
- **Source morphology:** masses/distance chosen for the cleanest legible WDM
  track + a meaningful SNR. A near-merger stellar-mass / low-IMBH system that
  visibly chirps across the observation (true ~30 Msun chirps too slowly over
  1 yr). Final numbers tuned during prototyping.
- **Sampled parameters (the "few"):** `tc` (coalescence time — places the track
  against the confusion modulation), `Mc` (chirp mass — sets the chirp track and
  time-in-band), `ln dL` (distance / SNR). Fix spins, sky position, inclination,
  polarization, eta (or sample eta as a 4th if mixing allows). 3 sampled params.
- **Compute:** prototype CPU smoke locally -> validate -> OzStar head-node CUDA.
  `ripplegw` behind the existing `[lisa]` optional extra. Colab-friendly entry.

## The two cases compared (same injection, same data)

| | Case A — stationary (the "wrong" model) | Case B — time-varying (ours) |
|---|---|---|
| Noise model | `log_psplines` stationary 1D log-P-spline PSD | WDM tensor-product log-P-spline surface `S(t,f)` |
| Source rep. | frequency-domain `ripple` waveform | WDM-transformed time-domain waveform |
| Likelihood | frequency-domain Whittle, one year-averaged PSD | WDM Whittle on the surface |
| Sampler | joint NUTS over (PSD coeffs, theta) | joint NUTS / blocked Gibbs over (surface, theta) |
| Expected result | biased / mis-calibrated theta posterior | calibrated theta posterior containing truth |

## Work items

1. **`datasets/sobh.py`** — mirror `datasets/lisa_tdi.py` discipline:
   - `sobh_waveform_fd(freqs, params)` — ripple IMRPhenomD `h+, hx` (jax).
   - `lisa_lw_response(t, sky, psi)` -> `F+(t), Fx(t)` long-wavelength antenna.
   - `sobh_strain_td(n, dt, params)` — projected, response-applied time series
     for the WDM branch; and the FD strain for the stationary branch.
   - `sobh_optimal_snr(...)` reusing the matched-filter convention already in
     `lisa_tdi.py` (TDI gen-2 noise convention preserved).
   - `_require_lisa`-style import guard (now also covers `ripplegw`).
2. **Differentiable joint source+noise sampler** — extend `tv_pspline_psd/joint.py`:
   - WDM branch: a `_sobh_signal` numpyro model that calls ripple inside the jax
     graph (theta -> h(theta) -> WDM coeffs) and reuses the whitened P-spline
     noise prior. New `run_joint_sobh_noise_mcmc` (single NUTS trajectory) and a
     blocked-Gibbs variant reusing the existing noise block. Fulfils the
     `TODO` at joint.py:204 (nonlinear source block).
   - This replaces the previous plan's gradient-free RWM block entirely.
3. **Stationary baseline** — thin adapter over `log_psplines` for Case A's
   year-averaged PSD + a frequency-domain Whittle joint fit over theta.
4. **`studies/sobh_tvpsd/`** — comparison driver (this dir): `--quick` CPU smoke
   and full scale, writes npz + figures. Mirrors `studies/lisa_gb/` layout.
   `submit.sbatch` for OzStar CUDA. `smoke.py` for fast correctness.
5. **`notes/scripts/make_sobh_demo.py`** + manuscript subsection — publication
   figure: WDM chirp-track surface, recovered TV-PSD, theta corner (Case A vs
   Case B), and coverage over `tc` at a confusion max vs min.

## Case A: frequency-domain stationary baseline (decided)

Per the user, Case A is built **literally in the frequency domain** with the
stationary PSD estimated by `log_psplines` (the nz-gravity multivariate study,
`pip` name `log_psplines`). Concrete integration path (scoped):

1. `MultivariateTimeseries(y=signal+noise, t=...)` (univariate `y.shape==(n,)`
   is supported; note it standardises internally -> carries a `scaling_factor`).
2. `make_pipeline(ts, PipelineConfig(n_knots, n_warmup, n_samples, ...)).run()`
   -> `PipelineResult.idata` (groups: posterior `weights_delta_0`/`phi_delta_0`,
   `spline_model` `diag_0_knots`/`diag_0_grid_points`, observed `periodogram`).
3. Extract the posterior stationary PSD via the package's own reconstruction
   helpers in `log_psplines.plotting.psd_matrix` (`*_from_idata -> EmpiricalPSD`)
   rather than reimplementing weights x basis; undo the standardisation scaling.
4. FD Whittle joint fit for `dL`: `d(f) ~ CN(h(f; dL), S_stat(f))` with the
   differentiable FD ripple waveform, NUTS over `dL` (init at truth). Compare the
   `dL` posterior (bias + width) against the WDM TV-PSD case (item 5).

The contrast the figure makes: a merger placed at a Galactic-confusion **max**
vs **min** epoch; the year-averaged stationary PSD mis-estimates the local noise
there, biasing/mis-calibrating `dL`, while the WDM TV-PSD is calibrated.

### Item 6 (FD stationary baseline): DONE and validated

`studies/sobh_tvpsd/fd_baseline.py` + `smoke_fd.py`:
- `estimate_stationary_psd` drives log_psplines (MultivariateTimeseries ->
  make_pipeline -> run), reconstructs the diagonal PSD from posterior
  `weights_delta_0` via `spline_model.compute_psd_quantiles`, interpolates to the
  rfft grid, and calibrates to the one-sided continuous-FT convention with
  `K = median(P1/S_lp)/ln2` (de-biases the exponential-periodogram median;
  truth-free). Recovers the physical instrument PSD to ~3% (ratio 1.03).
- `fd_dL_posterior`: since only dL is sampled and the amplitude scales as 1/dL,
  the FD template is exactly `(d_ref/dL) h_ref(f)` (no waveform regen in-sampler).
  NUTS Whittle over ln dL recovers injected dL at z~+1 against the stationary PSD.

Both inference branches are now validated end-to-end: WDM TV-PSD
(`run_joint_sobh_wdm_mcmc`) and FD stationary (`fd_baseline`). Remaining:
the comparison driver (inject one SOBH at a confusion max vs min, run both,
contrast the dL posterior + recovered noise), the publication figure, the
manuscript subsection, and OzStar/Colab glue.

## Item 7 (the demonstration): DONE

Two routes were built:
- `studies/sobh_tvpsd/comparison.py` -- the 2x2 with *estimated* PSDs
  (log_psplines FD vs WDM surface). Result was null: the estimated noise models
  inflate the dL error ~14x and mask the effect (a real finding about
  estimation-limited regimes).
- `studies/sobh_tvpsd/analytic_comparison.py` -- the **chosen** route: known
  PSD, closed-form dL amplitude weighted by the true S(t,f) [TV] vs the time
  average <S>_t(f) [stationary], over an ensemble, reporting 90% coverage.

Result (`notes/figures/sobh_tvpsd_analytic.png`):
- **Case A (stationary noise): TV and stationary agree**, both ~0.90 coverage.
- **Case B (non-stationary noise): TV stays calibrated (~0.90), stationary is
  overconfident and under-covers (~0.86, quoted err 3.5% vs actual 3.9%)**,
  worsening with a denser confusion foreground (0.83 at conf_boost x8).

Key physics finding (important for the paper framing): a broadband chirp's SNR
is *suppressed* where confusion dominates (SNR^2 = g^2/S), so it largely avoids
the confusion band -> the time-varying-confusion effect on parameters is
**real but modest** under nominal foreground (S_tv/S_stat ~ 1.3 at best, M~5e6),
and strengthens when the source SNR overlaps the modulated confusion (denser
foreground, or a time-localized burst at a confusion extreme, cf. the transient
script). Convention gotcha resolved: datasets/lisa.py uses a *digital* PSD
(E[w^2]=S); the FD matched-filter uses one-sided continuous-FT (1/Hz); they
differ by a factor 2*dt -- mixing them silently rescales the injected SNR.

## Validation gates (before scaling up)

- ripple SNR matches the analytic matched-filter SNR in the shared noise
  convention (the check `lisa_tdi.py` already does for the GB).
- Round-trip: FD ripple waveform -> time series -> WDM transform reproduces the
  injected chirp track on the surface.
- Smoke joint fit recovers injected theta on a **stationary**-noise control,
  then exhibits the Case-A bias on cyclostationary noise.
- jax gradients flow through ripple inside numpyro (no NaNs; NUTS adapts).

## Open risks / to verify during build

- ripple IMRPhenomD parameter vector + frequency-grid conventions (units of Mc,
  geometric vs SI, f_ref handling) — verify against ripple docs on first import.
- ripple FD lower-frequency cutoff / band coverage for mHz LISA frequencies.
- WDM grid resolution (nt, nf) vs the chirp slope — the track must be resolved.
- Long-wavelength response validity at the upper end of the LISA band for the
  chosen masses (acceptable for a demo; note the approximation in the text).

## Progress log

- **Item 1 + first validation gate: DONE.** `datasets/sobh.py` written and
  validated by `studies/sobh_tvpsd/smoke.py`:
  - ripple API confirmed: `ripplegw.waveforms.IMRPhenomD.gen_IMRPhenomD_hphc(f,
    theta8, f_ref)`, `theta8 = [Mc, eta, chi1, chi2, D(Mpc), tc, phic, inc]`.
  - FFT-embedding convention round-trips to machine precision (rel-err 1.5e-7);
    direct == round-trip SNR.
  - LW antenna built constructively from analytic spacecraft orbits; sky-average
    `<F+^2+Fx^2> = 0.317` matches the analytic single-Michelson LISA value
    `(sqrt3/2)^2 * 2/5 ~ 0.30`.
  - WDM transform shows a rising chirp track (mean-freq vs time corr +0.45).
  - ripplegw added to the `[lisa]` extra; `datasets/__init__.py` re-exports.

### Findings that shape items 2-5

- **jax WDM backend is feasible:** `wdm_transform` exposes
  `Backend(name, xp, fft)` + `register_backend`; only numpy is built in, but a
  jax backend (`xp=jax.numpy, fft=jax.numpy.fft`) can likely be registered to
  make the WDM forward transform differentiable -> WDM branch stays all-NUTS.
  Verify no numpy-only ops / in-place mutation in the WDM code path. Fallback if
  it doesn't work cleanly: WDM is linear, so precompute the signal's WDM via a
  fixed linear operator and differentiate only the projection, or run the WDM
  signal block gradient-free.
- **WDM-branch joint sampler (item 5): DONE and validated.** `_sobh_wdm_model`
  + `run_joint_sobh_wdm_mcmc` in `joint.py`; `studies/sobh_tvpsd/smoke_sampler.py`
  recovers (Mc, tc, dL) at smoke scale with 0 divergences. The jax WDM transform
  is differentiable + jit-able (confirmed), so the WDM branch is fully NUTS.
- **Demo-shaping findings from the sampler:**
  - The GW chirp-phase posterior is *extraordinarily* sharp and curved in Mc/tc
    (a 1% Mc error costs dchi2 ~ 6e4). NUTS must be **initialised at the
    maximum-likelihood point** (an 8%-offset start freezes near init). Standard
    for an injection study: we inject a known source and compare posterior
    width/bias between noise models, not do blind search.
  - **Mc and tc are phase-driven and noise-model-insensitive** (pinned to ~1e-7
    relative precision regardless of the PSD). The noise-sensitive parameter is
    **dL (distance/amplitude)** -- that is where stationary-vs-TV-PSD bias and
    mis-calibration appear. So the demonstration samples/compares **dL**, and
    uses **tc as an injection-placement knob** (merger at a confusion max vs
    min), mirroring `make_lisa_transient.py`. The sampler keeps Mc/tc samplable
    (init at truth) but the figure emphasis is dL + the recovered surface.
- **Tuning for the demo (defer to item 4/5):** default injection SNR is ~700 at
  D=3 Gpc -> dial distance to a target SNR (~20-50). The merger-dominated IMBH
  gives nf~3e4 WDM channels and a track that is somewhat transient-heavy;
  restrict to the source's active band and/or tune masses/window for the
  cleanest diagonal-track figure.
