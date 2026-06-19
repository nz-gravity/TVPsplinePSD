# LISA galactic-binary study (stationary row)

Frequency-domain vs WDM-domain Bayesian parameter estimation of a resolved
chirping galactic binary in **known stationary** LISA instrument noise. Per seed
the source `(f0, fdot, A, phi0)` is inferred with NumPyro/NUTS in both the
frequency-domain Whittle likelihood and the WDM-domain likelihood, using the
linearized Cartesian-amplitude model: sample `(z_f0, z_fdot, z_gc, z_gs)` with
`g_c = A cos(phi0)`, `g_s = A sin(phi0)`, so the strain is exactly bilinear in the
amplitudes. The two domains should give the same posterior (small freq-vs-WDM
JSD) — this is the **stationary-noise consistency check** of the manuscript's
LISA section (the top row of the noise-stationary / non-stationary grid).

## Provenance

`lisa_common.py` and `lisa_gb_study.py` are vendored, with light adaptation, from
`pywavelet/wdm_transform` (`docs/studies/lisa/`). They are the reference
implementation behind the `pywavelet/manuscript` GB demo.

## Run

```bash
# one seed, full production settings (slow: 1-yr grid, 1500 warmup x 2 chains x 2 domains)
python studies/lisa_gb/lisa_gb_study.py --seed 0
# population study + PP plot
python studies/lisa_gb/lisa_gb_study.py --nseeds 100
```

## Status

Vendored as the reference implementation; **not yet runnable here unmodified**.
It targets an older `jaxgb` waveform API (`get_kmin(params[None, 0:1])` and a
different parameter stacking); our installed `jaxgb` is 0.2.1, whose API is
`get_kmin(f0)` with an 8-vector `get_tdi(params, ...)`. The fix is bounded: the
waveform-embedding layer (`_gb_params`, `gb_full_rfft`, `gb_full_rfft_np`) must be
re-pointed at the 0.2.1 API, exactly as `datasets/lisa_tdi.py` already does
(`gb.get_kmin(f0=jnp.array([...]))`). Everything else (likelihoods, Cartesian
reparam, Fisher mass matrix, banding) is version-independent.

## TODO (manuscript 2x3 extension)

This provides the **stationary** row, frequency + WDM legs. Still to do:
- adapt the waveform layer to `jaxgb` 0.2.1 (above) and reproduce the
  freq-vs-WDM posterior overlap;
- a **moving short-time-Fourier (STFT)** leg (the phase-retaining R=2 front end;
  the power periodogram cannot subtract a coherent template);
- the **non-stationary** row: replace the known PSD with a jointly-inferred
  non-stationary noise surface via the blocked-Gibbs sampler
  (`run_gibbs_signal_noise_mcmc`), where the frequency-domain *stationary* Whittle
  is expected to bias while WDM and STFT recover.
