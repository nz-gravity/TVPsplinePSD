# WDM projection validity: compact support does not imply point evaluation

Status: revised 2026-08-13 after checking the production `nt=2048` M0 grid.
Scripts in `studies/wdm_projection_validity/`; the ESA production preflight is
`lisa_data_generation/ozstar/preflight_m0.py`.

## The question

Every fit in this repo models the expected WDM cell power by point-evaluating a
continuous PSD at the cell centre. The exact statement is

    E[w_nm^2] = \int |g_nm(f)|^2 S(f) df

and `S(f_m)` is only an approximation to it, valid when `S` is locally flat
across the atom's frequency support. Where `S` has a deep response null or a
steep power law, the integral samples a neighbourhood in which `S` is orders of
magnitude larger than at the centre, so the approximation would *underestimate*
and the data would look like it carried excess power.

This was raised as a candidate explanation for the A/E/T pilot's T-channel
excess (simulated/analytic ~20x in the counts-weighted mean below 0.3 mHz, ~1.2x
in the median, A and E clean; see `lisa_data_generation/`). The pattern fits the
hypothesis qualitatively: excess concentrated at nulls, largest where the PSD is
steepest, worst in T because T's nulls are deepest.

**It is not the explanation for the `nt=32` M1 pilot: the approximation is good
to ~6e-5 on that grid. It is not safe at deep response nulls on the coarser
`nt=2048` M0 grid, where the analytic reference must be projected through the
measured compact kernel.**

## 1. The frequency kernel is compactly supported

`wdm_transform` uses the cosine-tapered Meyer window with `a = 1/3`
(`windows.py:phi_unit`): flat over `|u| <= 1/3`, cosine taper to `|u| <= 2/3`,
identically zero beyond, where `u = (f - f_m) / df`.

Measured from the installed transform by injecting unit tones and reading
`sum_n w_nm^2` (this is the kernel of the code as implemented, discretisation
sidelobes included, with no window algebra):

| `\|u\|` | max response |
|---|---|
| < 0.5 | 1.0 |
| 0.5 - 0.7 | taper |
| 0.7 - 1.0 | 1.9e-24 |
| > 4 | 7.6e-24 |

Power fraction outside `|u| <= 2/3`: **3.3e-23**. Effective bandwidth
`sigma_W = 0.298 df`.

There are **no wings**. A cell sitting in a null cannot see power beyond 2/3 of
a channel spacing. This rules out long-range leakage, but it does not make
`S(f_m)` exact: local curvature anywhere inside that compact interval still
contributes to the WDM marginal power.

## 2. Predicted bias on the `nt=32` M1 grid is 1e-5

The A/E/T pilot ran at `nt = 32`, so `nf = 491520` and `df = 5.09e-7 Hz`. The
kernel half-width is `3.4e-7 Hz`; the analytic LISA PSD varies on mHz scales,
three decades wider. Computing `S_eff(m) / S(f_m)` against the real
`analytic_aet_noise_psd`, worst case over band, channel and epoch:

| grid | df [Hz] | A | E | T |
|---|---|---|---|---|
| nt=32 (as run) | 5.09e-7 | 4.1e-5 | 4.1e-5 | **6.1e-5** |
| nt=64 | 1.02e-6 | 8.2e-5 | 8.2e-5 | 1.2e-4 |
| nt=128 | 2.03e-6 | 1.7e-4 | 1.7e-4 | 2.5e-4 |

0.006% against a required 1900%.

## 3. The `nt=2048` M0 grid requires explicit projection

The year-long M0 analysis uses `df = 3.2552083e-5 Hz`, 64 times the M1 pilot's
spacing. The leading smooth-spectrum curvature term therefore grows by
`64^2 = 4096`: multiplying the measured `6.1e-5` M1 worst case gives roughly
`0.25`, or 25%, not 0.2%. More importantly, relative error at an exceptionally
deep point null is not controlled by a continuum-relative curvature estimate.

Direct evaluation on the actual X2 production grid at three epochs finds a
projected-to-point ratio as large as `3.58e3` at the deepest sampled null. A
16-node Gauss--Legendre integral of the squared installed Meyer window agrees
with a 32-node rule to a maximum relative change of `2.18e-4` at those cells.
The effect is therefore a converged local convolution, not a tail or
grid-luck artifact. Away from nulls the correction remains small; the earlier
audit found a median correction near 0.16%, and a maximum below 3 mHz near
0.30%.

M0 now uses the projected analytic OMS/TM reference and projected simulation
truth in inference and validation. Point-evaluated response surfaces are kept
only to define the diagnostic null corridors. The Galactic truth is projected
with zero power outside the frequency band used to generate its time series.

## 4. End-to-end `nt=32` recovery is unbiased, including in nulls

Stationary Gaussian noise drawn with the real analytic A/E/T PSD as input,
pushed through the exact transform and `2 dt / N` conversion, recovered vs input
(`nt=32`, `nf=65536`, per-cell sd `sqrt(2/30) = 0.258`):

| chan | selection | mean | median |
|---|---|---|---|
| T | 1e-4 - 1e-3 Hz | 0.992 | 0.979 |
| T | 1e-3 - 1e-2 Hz | 0.994 | 0.974 |
| T | null cells >3 nat deep | 0.989 | 0.980 |
| A | null cells >3 nat deep | 0.974 | 0.939 |

Unbiased to <=1.5% everywhere. The medians land at 0.974-0.980, which is exactly
`1 - 2/(3 n_time)`, the median/mean ratio of a `chi^2_30` -- so the pilot's ~1.2x
*median* excess is not a projection effect either.

## When point evaluation can be trusted

The approximation is safe because `df` is far finer than any structure in `S`.
The relevant number is the curvature over a kernel width; for a local power law
`S ~ f^alpha` the bias is

    S_eff / S(f_m) - 1  ~  (alpha (alpha - 1) / 2) (0.298 df / f_m)^2

Re-check before trusting `S(f_m)` if any of these change:

- **Large `nt`** (coarse frequency resolution). The bias grows as `df^2` and
  `df = 1/(2 nf dt) = nt/(2 N dt)`. The `nt=2048` M0 grid is the concrete case
  where explicit projection is now required.
- **The lowest retained channels.** The bias scales as `(df / f_m)^2`, i.e. as
  `1/m^2`. It is only ever large for `m` of order a few.
- **Genuinely narrow or exceptionally deep features.** Instrument lines,
  resolved Galactic binaries, and deep sampled TDI zeros can make the
  continuum-relative error small while the projected-to-point ratio is large.
- **A different window parameter `a`.** Support is `|u| <= 1 - a`; the compact
  support itself is a property of the Meyer construction and survives, but the
  bandwidth changes.

## Consequences for the M1 T-channel discrepancy

Projection remains ruled out as the cause on the `nt=32` M1 grid. In particular this
does *not* rehabilitate or retract anything about the fitted null-leakage
parameter: `kappa0` came out 80x from its predicted value with the wrong-signed
slope, and that conclusion was drawn from the parameter's own behaviour, not
from any window argument. It still stands.

Two candidates remain, neither tested here:

1. **Model vs simulation.** These checks used the analytic model as both input
   and reference, so they are blind to a mismatch between
   `AnalyticOMSNoiseModel`/`AnalyticTMNoiseModel` and how `noise2a/tdi.h5` was
   generated. Equal-arm and residual laser noise are already ruled out
   elsewhere, which points at the OMS/TM noise definitions themselves.
2. **The interpolated reference.** `noise_reference_binned` is interpolated in
   log-log from the 256-point `truth/frequency_hz` grid
   (`run_aet_diagonal_pilot.py:613`), whereas the transfer functions the fit
   actually uses are evaluated directly on the analysis grid (`:689`). If the
   20x was quoted against the interpolated reference, some of it is a diagnostic
   artifact rather than a fit artifact. Worth establishing which.

The discriminating measurement is the simulated T PSD from `tdi/noise` compared
against both references, now that the projection is cleared as a confounder.
