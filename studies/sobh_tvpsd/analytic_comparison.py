"""Analytic-PSD demonstration: time-varying vs stationary noise model.

The cleanest statement of the claim, with the noise PSD *known* (no estimation),
mirroring ``notes/scripts/make_lisa_transient.py``. The chirping source's WDM
coefficients ``g(t, f)`` are measured against per-cell noise variance ``S(t, f)``:

  * TV analysis      -- weight by the true ``S(t, f) = S_inst(f) + m(u)^2 S_gal(f)``.
  * Stationary       -- weight by the time average ``<S>_t(f) = S_inst + S_gal``.

Two datasets:
  Case A: noise drawn from the *stationary* variance (m == 1).
  Case B: noise drawn from the *non-stationary* variance (cyclostationary m(u)).

For a linear amplitude ``a = d_ref/dL`` the posterior is closed-form,
``a_hat = sum(g c / S) / sum(g^2 / S)``, ``var(a_hat) = 1 / sum(g^2 / S)`` under
the assumed ``S``. We report, over an ensemble of noise realizations, the bias,
the quoted vs actual scatter, and the 90% coverage of the truth -- the
calibration metric. Expectation:
  * Case A: TV and stationary agree and both cover ~90% (stationary truth).
  * Case B: with the merger at a Galactic-confusion *maximum*, the stationary
    model under-estimates the local noise -> overconfident -> under-covers, while
    the TV model stays calibrated.

    uv run python -m studies.sobh_tvpsd.analytic_comparison
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from datasets.lisa import (  # noqa: E402
    digman_cornish_power_modulation,
    lisa_galactic_confusion_psd,
    lisa_instrument_psd,
)
from datasets.sobh import SOBHParams  # noqa: E402
from tv_pspline_psd import (  # noqa: E402
    PSplineConfig,
    build_sobh_wdm_grid,
    make_sobh_wdm_signal_fn,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "notes" / "figures"
RES_DIR = Path(__file__).resolve().parent / "results"


def _amp_posterior(g, c, S):
    """Closed-form amplitude posterior under assumed per-cell variance ``S``."""
    w = 1.0 / S
    a_hat = np.sum(g * c * w) / np.sum(g**2 * w)
    var = 1.0 / np.sum(g**2 * w)
    return a_hat, np.sqrt(var)


def run(n_real: int = 1000, target_snr: float = 25.0, m_total: float = 5.0e6,
        conf_boost: float = 1.0) -> dict:
    dt, nt, n = 50.0, 32, 2**13
    cfg = PSplineConfig(trim_low_freq_channels=13, trim_high_freq_channels=77)
    T = n * dt

    params = SOBHParams(m1=m_total / 2, m2=m_total / 2)
    u = (np.arange(n) + 0.5) / n
    r = digman_cornish_power_modulation(u, channel="A", n_year_cycles=1.0)
    params.tc = float(u[np.argmax(r)] * T)  # merger at confusion maximum

    grid = build_sobh_wdm_grid(n, dt, nt, params, cfg)
    signal_fn = make_sobh_wdm_signal_fn(grid)
    g = np.asarray(signal_fn(jnp.asarray(
        [params.chirp_mass, params.tc, np.log(params.distance)])))  # (nt, nf)

    # Per-cell noise variance on the WDM grid (analytic E[w^2] target).
    s_inst = lisa_instrument_psd(grid.freq_grid)[None, :]
    s_gal = conf_boost * lisa_galactic_confusion_psd(grid.freq_grid)[None, :]
    m2 = digman_cornish_power_modulation(grid.time_grid, channel="A",
                                         n_year_cycles=1.0)[:, None]
    S_tv = s_inst + m2 * s_gal              # time-varying truth (Case B)
    S_stat = s_inst + s_gal                 # <m^2>=1 time average (stationary)
    S_stat = np.broadcast_to(S_stat, g.shape)

    cases = {"A_stationary": np.broadcast_to(S_stat, g.shape),
             "B_nonstationary": S_tv}
    # Scale the signal so the (TV) optimal SNR hits the target.
    snr0 = np.sqrt(np.sum(g**2 / S_tv))
    g = g * (target_snr / snr0)

    out = {"cases": {}, "snr": target_snr}
    rng = np.random.default_rng(0)
    for case, S_data in cases.items():
        a_tv, e_tv, a_st, e_st = [], [], [], []
        for _ in range(n_real):
            noise = rng.standard_normal(g.shape) * np.sqrt(S_data)
            c = g + noise  # truth amplitude a = 1
            ah, eh = _amp_posterior(g, c, S_tv); a_tv.append(ah); e_tv.append(eh)
            ah, eh = _amp_posterior(g, c, S_stat); a_st.append(ah); e_st.append(eh)
        a_tv, e_tv, a_st, e_st = map(np.asarray, (a_tv, e_tv, a_st, e_st))

        def _summ(a, e):
            cov = float(np.mean(np.abs(a - 1.0) < 1.645 * e))  # 90% interval
            return {"bias": float(a.mean() - 1.0), "quoted": float(e.mean()),
                    "actual": float(a.std()), "coverage90": cov, "a": a, "e": e}
        out["cases"][case] = {"TV": _summ(a_tv, e_tv), "stat": _summ(a_st, e_st)}
        for nm, key in (("TV-PSD", "TV"), ("stationary", "stat")):
            s = out["cases"][case][key]
            print(f"[{case}] {nm:10s} quoted-err={s['quoted']*100:.2f}%  "
                  f"actual-scatter={s['actual']*100:.2f}%  90%-coverage={s['coverage90']:.2f}")
    return out


def make_figure(out: dict) -> None:
    import matplotlib.pyplot as plt
    from tv_pspline_psd import save_figure, set_paper_style
    set_paper_style()
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: amplitude posteriors for the non-stationary case (the interesting one).
    c = out["cases"]["B_nonstationary"]
    for key, color, lab in (("stat", "tab:red", "Stationary PSD"),
                            ("TV", "tab:green", "Time-varying PSD")):
        ax_h.hist(c[key]["a"], bins=45, density=True, histtype="step", color=color,
                  lw=2, label=f"{lab}\n(quoted {c[key]['quoted']*100:.1f}%, "
                              f"actual {c[key]['actual']*100:.1f}%)")
    ax_h.axvline(1.0, color="0.3", ls="--", lw=1)
    ax_h.set_xlabel(r"$\hat a = d_{\rm ref}/d_L$"); ax_h.set_ylabel("density")
    ax_h.set_title("Case B: non-stationary noise"); ax_h.legend(frameon=False, fontsize=8)

    # Right: 90% coverage, both cases, both models (the calibration metric).
    cases = ["A_stationary", "B_nonstationary"]; labels = ["Case A\n(stationary)", "Case B\n(non-stat.)"]
    x = np.arange(2); w = 0.35
    ax_c.bar(x - w/2, [out["cases"][cs]["stat"]["coverage90"] for cs in cases], w,
             color="tab:red", label="Stationary PSD")
    ax_c.bar(x + w/2, [out["cases"][cs]["TV"]["coverage90"] for cs in cases], w,
             color="tab:green", label="Time-varying PSD")
    ax_c.axhline(0.90, color="0.3", ls="--", lw=1, label="nominal 90%")
    ax_c.set_xticks(x); ax_c.set_xticklabels(labels); ax_c.set_ylim(0.7, 1.0)
    ax_c.set_ylabel("90% interval coverage"); ax_c.set_title("Calibration")
    ax_c.legend(frameon=False, fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIG_DIR / "sobh_tvpsd_analytic.png")
    print(f"[figure] wrote {FIG_DIR / 'sobh_tvpsd_analytic.png'}")


def main() -> None:
    out = run()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(out)


if __name__ == "__main__":
    main()
