"""Case A: frequency-domain stationary baseline (log_psplines PSD + Whittle).

The stationary noise PSD is estimated with ``log_psplines`` (the nz-gravity
multivariate log-P-spline study) and the source distance ``dL`` is inferred with
a frequency-domain Whittle likelihood. This is the deliberately *stationary*
counterpart to the WDM time-varying-PSD fit: it sees one time-averaged spectrum,
so a source whose SNR accumulates at a Galactic-confusion extreme is measured
against the wrong local noise.

Conventions. We work in the one-sided continuous-FT convention of
``datasets.sobh`` / ``datasets.lisa_tdi``: ``h(f) = dt * rfft(x)``, one-sided PSD
``S(f)`` in 1/Hz, matched-filter ``SNR^2 = 4 df sum_{f>0} |h|^2 / S``. The
``log_psplines`` PSD is returned in its own internal convention, so we rescale it
to this one by the constant ``K = median(P1 / S_lp)`` over noise-dominated bins,
where ``P1(f) = (2 dt / n) |rfft(x)|^2`` is the one-sided periodogram (a
truth-free, data-derived calibration).

Because only ``dL`` is sampled and the source amplitude scales as ``1/dL``, the
frequency-domain signal model is exactly ``h(f; dL) = (d_ref / dL) h_ref(f)``
with ``h_ref`` the (fully responded) injected waveform at a reference distance --
no waveform regeneration inside the sampler.
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402


def estimate_stationary_psd(
    data: np.ndarray, dt: float, *, n_knots: int = 30,
    n_warmup: int = 300, n_samples: int = 300, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Stationary one-sided PSD on the rfft grid via log_psplines.

    Returns ``{freq, S_stat, S_lp, fgrid_lp, K}``: ``S_stat`` is the posterior-
    median PSD interpolated to ``rfftfreq(n, dt)`` and rescaled to the one-sided
    continuous-FT convention; ``K`` is the calibration constant.
    """
    from log_psplines import make_pipeline
    from log_psplines.datatypes import MultivariateTimeseries
    from log_psplines.pipeline import PipelineConfig

    n = len(data)
    ts = MultivariateTimeseries(y=np.asarray(data, dtype=float),
                                t=np.arange(n) * dt)
    cfg = PipelineConfig(n_warmup=n_warmup, n_samples=n_samples, n_knots=n_knots,
                         verbose=False, outdir=None, init_from_vi=False,
                         rng_key=seed)
    pipe = make_pipeline(ts, cfg)
    res = pipe.run()

    sm = pipe.spline_model
    w = np.asarray(res.idata.posterior["weights_delta_0"])[0]  # (ns, n_basis)
    basis = np.asarray(sm.component_specs[sm.delta_key(0)].model.basis)
    log_delta_sq = jnp.asarray((w @ basis.T)[:, :, None])      # (ns, N, 1)
    empty = jnp.zeros((w.shape[0], basis.shape[0], 0))
    pr, _, _ = sm.compute_psd_quantiles(log_delta_sq, empty, empty, n_samples_max=120)
    s_lp = np.asarray(pr)[1, :, 0, 0]                          # median model PSD
    fgrid_lp = np.asarray(pipe.data.freq)

    freq = np.fft.rfftfreq(n, d=dt)
    s_lp_rfft = np.interp(freq, fgrid_lp, s_lp, left=s_lp[0], right=s_lp[-1])
    # Calibrate to the one-sided continuous-FT convention via the periodogram.
    # P1 ~ Exponential(S) per interior bin, so median(P1/S_lp) = ln2 * (S/S_lp);
    # dividing by ln2 de-biases the level, and the median is robust to the few
    # signal-dominated bins.
    p1 = (2.0 * dt / n) * np.abs(np.fft.rfft(np.asarray(data)))**2
    interior = (freq > 0) & (freq < freq[-1])
    k = float(np.median(p1[interior] / np.maximum(s_lp_rfft[interior], 1e-300))
              / np.log(2.0))
    return {"freq": freq, "S_stat": k * s_lp_rfft, "S_lp": s_lp,
            "fgrid_lp": fgrid_lp, "K": k}


def _fd_dL_model(d_re, d_im, h_re, h_im, inv_s_2df, d_ref, ln_dl_ref, ln_dl_scale):
    """Whittle likelihood for dL with a fixed responded template ``h_ref``.

    The signal is ``(d_ref / dL) * h_ref``; the (one-sided) Gaussian log-density
    is ``-2 df sum_{f>0} |d - h(dL)|^2 / S``. ``inv_s_2df = 2 df / S``.
    """
    z = numpyro.sample("z_d", dist.Normal(0.0, 1.0))
    ln_dl = ln_dl_ref + ln_dl_scale * z
    amp = d_ref / jnp.exp(ln_dl)
    r_re = d_re - amp * h_re
    r_im = d_im - amp * h_im
    numpyro.factor("whittle_fd", -jnp.sum((r_re**2 + r_im**2) * inv_s_2df))
    numpyro.deterministic("dL", jnp.exp(ln_dl))


def fd_dL_posterior(
    data: np.ndarray, h_ref_td: np.ndarray, dt: float, *, S_stat: np.ndarray,
    d_ref: float, dl_ref: float, dl_scale: float = 0.5,
    n_warmup: int = 500, n_samples: int = 1000, seed: int = 1,
) -> dict[str, np.ndarray]:
    """Frequency-domain Whittle posterior on dL against a stationary PSD.

    Args:
        data: time series (signal + noise).
        h_ref_td: fully responded injected waveform at distance ``d_ref`` (time
            domain); the FD template is ``dt * rfft(h_ref_td)``.
        S_stat: one-sided PSD on ``rfftfreq(n, dt)`` (from ``estimate_stationary_psd``).
        d_ref, dl_ref: reference distance of ``h_ref_td`` and the prior centre.
    """
    n = len(data)
    freq = np.fft.rfftfreq(n, d=dt)
    df = 1.0 / (n * dt)
    d_fd = dt * np.fft.rfft(np.asarray(data))
    h_fd = dt * np.fft.rfft(np.asarray(h_ref_td))
    mask = freq > 0
    inv_s_2df = 2.0 * df / S_stat[mask]

    mcmc = MCMC(NUTS(_fd_dL_model, target_accept_prob=0.9),
                num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed),
             jnp.asarray(d_fd[mask].real), jnp.asarray(d_fd[mask].imag),
             jnp.asarray(h_fd[mask].real), jnp.asarray(h_fd[mask].imag),
             jnp.asarray(inv_s_2df), float(d_ref), float(np.log(dl_ref)),
             float(dl_scale))
    s = mcmc.get_samples()
    return {"dL": np.asarray(s["dL"])}
