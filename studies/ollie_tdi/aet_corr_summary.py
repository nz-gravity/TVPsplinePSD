"""AET residual cross-correlation vs frequency, overlaid across window lengths.

Coefficient-level companion to ``mojito_validation.aet_cross_channel``: the
Pearson correlation of WDM coefficients is invariant to any per-channel positive
scaling, so both the spectral shape S(f) and any common power drift cancel and
``corr(w_A, w_E)`` in a narrow frequency bin equals the whitened
``corr(z_A, z_E)`` -- no MCMC fit required. That makes it cheap enough to span
every window length (1 week ... full ~2 yr) and overlay them on one figure.

Each channel is still normalised per frequency channel by its time-RMS (an exact
empirical whitening for stationary noise; harmless for the correlation otherwise)
before binning, so heteroscedasticity across channels within a bin can't tilt the
pooled correlation.

Run:
    python studies/ollie_tdi/aet_corr_summary.py
    python studies/ollie_tdi/aet_corr_summary.py --only 1_week 1_month 6_month
"""

from __future__ import annotations

import argparse
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from check_XYZ import DATA, DT, orthogonal_aet
from fit_mojito_segment import band_trims
from mojito_experiments import EXP_DIR, EXPERIMENTS, START_DAY
from scipy.stats import pearsonr

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients

set_paper_style()

FMIN, FMAX = 1e-4, 1e-1
PAIRS = [("A", "E"), ("A", "T"), ("E", "T")]
ORDER = ["1_week", "1_month", "6_month", "1_year", "1.5_year", "full"]


def load_aet(nt: int, days):
    """Load A/E/T over the experiment window (WDM-valid length)."""
    with h5py.File(DATA, "r") as h:
        grp = h["processed/segment0"]
        n_full = grp["X"].shape[0]
        t_full = n_full * DT / 86400.0
        if days is None:
            i0, span = 0, t_full
        else:
            i0 = int(round(START_DAY * 86400.0 / DT))
            span = days
        nf = int(round(span * 86400.0 / DT)) // nt
        nf -= nf % 2
        n_use = nt * nf
        xyz = {c: grp[c][i0 : i0 + n_use] for c in ("X", "Y", "Z")}
    return orthogonal_aet(xyz), n_use


def corr_vs_freq(exp: str, n_fbins: int = 28) -> dict:
    cfg = EXPERIMENTS[exp]
    nt = cfg["nt"]
    aet, n_use = load_aet(nt, cfg["days"])
    trim_low, trim_high = band_trims(n_use, nt, FMIN, FMAX)
    config = PSplineConfig(trim_time_bins=2, trim_low_freq_channels=trim_low,
                           trim_high_freq_channels=trim_high)
    z, fg = {}, None
    for ch in "AET":
        coeffs, _, fg = wdm_analysis_coefficients(aet[ch], DT, nt, config)
        c = coeffs[0] if coeffs.ndim == 3 else coeffs
        z[ch] = c / np.sqrt((c**2).mean(0, keepdims=True))  # per-channel time-RMS whitening

    edges = np.logspace(np.log10(fg[0]), np.log10(fg[-1]), n_fbins + 1)
    fc = np.sqrt(edges[:-1] * edges[1:])
    idx = np.clip(np.digitize(fg, edges) - 1, 0, n_fbins - 1)
    r_of_f = {f"{i}{j}": np.full(n_fbins, np.nan) for i, j in PAIRS}
    nbin = np.zeros(n_fbins, dtype=int)
    for b in range(n_fbins):
        m = idx == b
        nbin[b] = int(m.sum()) * z["A"].shape[0]
        if m.sum() < 1:
            continue
        for i, j in PAIRS:
            r_of_f[f"{i}{j}"][b] = pearsonr(z[i][:, m].ravel(), z[j][:, m].ravel())[0]
    overall = {f"{i}{j}": float(pearsonr(z[i].ravel(), z[j].ravel())[0]) for i, j in PAIRS}
    print(f"[{exp}] n_time={z['A'].shape[0]} overall " +
          " ".join(f"{k}={v:+.3f}" for k, v in overall.items()))
    return dict(fc=fc, nbin=nbin, r_of_f=r_of_f, overall=overall,
                days=(cfg["days"] if cfg["days"] else n_use * DT / 86400.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", default=ORDER, choices=ORDER)
    args = p.parse_args()
    names = [e for e in ORDER if e in args.only]
    results = {e: corr_vs_freq(e) for e in names}

    cmap = plt.cm.viridis(np.linspace(0.0, 0.88, len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), constrained_layout=True, sharey=True)
    for ax, (i, j) in zip(axes, PAIRS):
        key = f"{i}{j}"
        for c, e in zip(cmap, names):
            r = results[e]
            ax.plot(r["fc"], r["r_of_f"][key], color=c, lw=1.1,
                    label=f"{e.replace('_',' ')} ({r['days']:.0f} d)")
        # +/-3 sigma null band from the largest sample (tightest); annotate typical
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("f [Hz]")
        ax.set_title(f"corr$(z_{i}, z_{j})$", fontsize=10)
    axes[0].set_ylabel(r"residual cross-correlation")
    axes[0].set_ylim(-0.25, 0.25)
    axes[-1].legend(fontsize=6.5, title="window", loc="upper left")
    fig.suptitle("AET cross-channel correlation vs frequency, by segment length", fontsize=10)
    out = EXP_DIR / "aet_corr_vs_freq_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)

    with open(EXP_DIR / "aet_corr_summary.json", "w") as fp:
        json.dump({e: {"days": r["days"], "overall": r["overall"],
                       "fc": r["fc"].tolist(),
                       "r_of_f": {k: np.where(np.isnan(v), None, v).tolist()
                                  for k, v in r["r_of_f"].items()}}
                   for e, r in results.items()}, fp, indent=2)
    print(f"[done] overlay -> {out}")


if __name__ == "__main__":
    main()
