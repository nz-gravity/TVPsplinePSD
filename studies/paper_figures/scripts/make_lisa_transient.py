r"""Transient-source corner: stationary frequency-domain Whittle vs WDM P-spline.

A year-long Galactic binary is robust to the stationarity assumption (it averages
over the annual confusion modulation; see the ensemble). A *time-localized*
source -- a burst, or the final days of a massive-black-hole-binary inspiral --
accumulates its SNR at one epoch and is measured against the noise *there*, which
the cyclostationary Galactic foreground swings by ~7x over the year.

For this source-only benchmark we model its complex amplitude directly in one
localized frequency channel over time:

    c_n = A exp(i[phi0 + 2 pi df t_n]) W(u_n) + noise_n,   noise_n ~ CN(0, S(u_n,f0))

with W a localized (transient) envelope. We infer (f0, A, phi0) with NUTS twice on
the same data: once with the **stationary frequency-domain Whittle** noise model
(a single time-averaged PSD) and once with the **WDM P-spline time-varying PSD**
(the per-epoch noise our estimator provides). For a transient at a confusion
maximum the stationary analysis is overconfident -- a tight posterior displaced
from the truth -- while the time-varying analysis is calibrated and contains it.

Saves ``studies/paper_figures/figures/lisa_transient_corner.png`` and prints the 90% coverage of
each model over many realizations.

    uv run python studies/paper_figures/scripts/make_lisa_transient.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from tv_pspline_psd import save_figure, set_paper_style  # noqa: E402
from tv_pspline_psd.datasets import digman_cornish_power_modulation  # noqa: E402
from tv_pspline_psd.datasets.lisa_tdi import (  # noqa: E402
    ae_tdi_confusion_psd,
    ae_tdi_noise_psd,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
_YEAR = 365.25 * 86400.0
F0 = 1.5e-3
NT = 512          # localized time bins over the year
TAU = 8.0         # transient envelope width, in segments
SNR = 16.0        # optimal SNR against the true local noise
PHI0_TRUE = 0.6


def local_noise_variance():
    u = (np.arange(NT) + 0.5) / NT
    s_inst = float(ae_tdi_noise_psd(np.array([F0]), channel="A")[0])
    s_conf = float(ae_tdi_confusion_psd(np.array([F0]), channel="A")[0])
    r = digman_cornish_power_modulation(u, channel="A", n_year_cycles=1.0)
    return u, s_inst + r * s_conf


def _model(c_re, c_im, W, u, sig2, amp_scale, df_scale):
    z_gc = numpyro.sample("z_gc", dist.Normal(0.0, 3.0))
    z_gs = numpyro.sample("z_gs", dist.Normal(0.0, 3.0))
    z_df = numpyro.sample("z_df", dist.Normal(0.0, 1.0))
    gc, gs, df = amp_scale * z_gc, amp_scale * z_gs, df_scale * z_df
    ph = 2.0 * jnp.pi * df * u
    m_re = W * (gc * jnp.cos(ph) - gs * jnp.sin(ph))
    m_im = W * (gc * jnp.sin(ph) + gs * jnp.cos(ph))
    numpyro.factor("like", -jnp.sum(((c_re - m_re) ** 2 + (c_im - m_im) ** 2) / sig2))
    numpyro.deterministic("A", jnp.hypot(gc, gs))
    numpyro.deterministic("phi0", jnp.arctan2(gs, gc))
    numpyro.deterministic("df", df)


def _run(c, W, u, sig2_model, amp_scale, df_scale, seed):
    mcmc = MCMC(NUTS(_model, target_accept_prob=0.9), num_warmup=800, num_samples=2000,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(c.real), jnp.asarray(c.imag),
             jnp.asarray(W), jnp.asarray(u), jnp.asarray(sig2_model),
             float(amp_scale), float(df_scale))
    s = mcmc.get_samples()
    return {"A": np.asarray(s["A"]), "phi0": np.asarray(s["phi0"]), "df": np.asarray(s["df"])}


def _draw(W, u, sig2, A_true, rng):
    """One complex-coefficient realization of the transient + local noise."""
    sig = (A_true * np.exp(1j * PHI0_TRUE)) * W  # df_true = 0 (source at f0)
    noise = (rng.standard_normal(NT) + 1j * rng.standard_normal(NT)) * np.sqrt(sig2 / 2.0)
    return sig + noise


def _coverage(W, u, sig2, amp_scale, A_true, n_real=300):
    """Fraction of realizations whose 90% (A, phi0) region contains the truth,
    analytically (g_c, g_s are exactly Gaussian)."""
    g_true = np.array([A_true * np.cos(PHI0_TRUE), A_true * np.sin(PHI0_TRUE)])
    rng = np.random.default_rng(0)
    out = {}
    for name, s2 in (("nonstat", sig2), ("stat", np.full(NT, sig2.mean()))):
        F = np.sum(W**2 / s2)
        noise = (rng.standard_normal((n_real, NT)) + 1j * rng.standard_normal((n_real, NT))) \
            * np.sqrt(sig2 / 2.0)
        c = (g_true[0] + 1j * g_true[1]) * W[None, :] + noise
        ghat = (c @ (W / s2)) / F
        d2 = ((ghat.real - g_true[0]) ** 2 + (ghat.imag - g_true[1]) ** 2) / (0.5 / F)
        out[name] = float(np.mean(d2 < 4.605170))
    return out


def main() -> None:
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    u, sig2 = local_noise_variance()
    # Transient at a confusion minimum: the quiet epoch the time-varying PSD can
    # exploit but a stationary (year-averaged) PSD cannot -- the WDM posterior is
    # ~2x tighter and calibrated, the stationary one needlessly wide (and, at a
    # confusion maximum, overconfident; see the coverage printed below).
    itr = int(np.argmin(sig2))
    W = np.exp(-0.5 * ((np.arange(NT) - itr) / TAU) ** 2)
    A_true = SNR / np.sqrt(np.sum(W**2 / sig2))
    amp_scale = A_true
    df_scale = 1.0  # df in cycles over the (rescaled) observation

    cov = _coverage(W, u, sig2, amp_scale, A_true)
    Wpk = np.exp(-0.5 * ((np.arange(NT) - int(np.argmax(sig2))) / TAU) ** 2)
    cov_pk = _coverage(Wpk, u, sig2, amp_scale,
                       SNR / np.sqrt(np.sum(Wpk**2 / sig2)))
    print(f"[coverage] TROUGH stationary={cov['stat']:.2f} non-stationary={cov['nonstat']:.2f}")
    print(f"[coverage] PEAK   stationary={cov_pk['stat']:.2f} non-stationary={cov_pk['nonstat']:.2f}")
    print(f"[precision] stationary/non-stationary posterior width (trough) "
          f"= {np.sqrt(sig2.mean()/sig2[int(np.argmin(sig2))]):.1f}x")

    rng = np.random.default_rng(3)
    c = _draw(W, u, sig2, A_true, rng)
    post = {
        "stat": _run(c, W, u, np.full(NT, sig2.mean()), amp_scale, df_scale, 1),
        "nonstat": _run(c, W, u, sig2, amp_scale, df_scale, 1),
    }

    # Build the (f0, A, phi0) sample matrices.
    import corner
    import matplotlib.pyplot as plt

    labels = [r"$\Delta f_0$ [bins]", r"$A/A_{\rm true}$", r"$\phi_0$ [rad]"]
    truths = [0.0, 1.0, PHI0_TRUE]

    def _mat(p):
        return np.column_stack([p["df"], p["A"] / A_true, p["phi0"]])

    colors = {"stat": "tab:red", "nonstat": "tab:green"}
    legend = {"stat": "Stationary Whittle (frequency domain)",
              "nonstat": "WDM P-spline (time-varying PSD)"}
    styles = {"stat": "dashed", "nonstat": "solid"}
    fig = None
    for name in ("stat", "nonstat"):
        fig = corner.corner(_mat(post[name]), labels=labels, truths=truths, fig=fig,
                            color=colors[name], truth_color="0.35", levels=(0.5, 0.9),
                            plot_datapoints=False, plot_density=False, fill_contours=False,
                            hist_kwargs={"density": True, "linestyle": styles[name]},
                            contour_kwargs={"linestyles": styles[name]})
    fig.legend(handles=[plt.Line2D([0], [0], color=colors[n], ls=styles[n], label=legend[n])
                        for n in ("stat", "nonstat")], loc="upper right", frameon=False, fontsize=11)
    save_figure(fig, FIG_DIR / "lisa_transient_corner.png")
    print(f"[figure] wrote {FIG_DIR / 'lisa_transient_corner.png'}")


if __name__ == "__main__":
    main()
