"""Raw-WDM linear-frequency zoom on the first Michelson null, 30-day dataset.

Companion to null_zoom.py (which targets the 2-year Mojito data): shows the
null of the X2 channel drifting along the parameter-free 1/(2<L(t)>)
prediction in the raw coefficients, before any fitting. <L> averages the four
arms adjacent to S/C 1 (12, 21, 13, 31).

Run:
    python studies/ollie_tdi/null_zoom_30day.py
"""

from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients

from fit_aet_fullband import DATA_FULL, DECIMATE, RESULTS_DIR, fft_decimate

set_paper_style()

FBAND = (0.0585, 0.0625)


def main() -> None:
    with h5py.File(DATA_FULL) as h:
        x = h["tdis/X2"][:]
        ltts = np.stack([h[f"ltts/ltt_{k}"][:] for k in ("12", "21", "13", "31")])
    L_bar = ltts.mean(axis=0)

    data = fft_decimate(x, DECIMATE)
    dt = 0.25 * DECIMATE
    nt = 128
    config = PSplineConfig(trim_time_bins=2, trim_low_freq_channels=1)
    coeffs, time_grid, freq_grid = wdm_analysis_coefficients(data, dt, nt, config)
    t_obs_d = data.size * dt / 86400
    tdays = time_grid * t_obs_d
    logp = np.log(coeffs**2)

    lo, hi = FBAND
    band = (freq_grid >= lo) & (freq_grid <= hi)
    # Light box smoothing tames the per-cell chi^2 speckle (cf. null_zoom.py).
    logp_s = uniform_filter(logp[:, band], size=(3, 3))
    vmin, vmax = np.percentile(logp_s, [2.0, 99.8])

    t_L = np.linspace(0, t_obs_d, L_bar.size)
    fnull = 1.0 / (2.0 * np.interp(tdays, t_L, L_bar))

    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    mesh = ax.pcolormesh(tdays, freq_grid[band], logp_s.T, shading="auto",
                         cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot(tdays, fnull, "r-", lw=1.2, label=r"$1/2\langle L\rangle$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("f [Hz]")
    fig.colorbar(mesh, ax=ax, shrink=0.9, label=r"$\log |w|^2$")

    out = RESULTS_DIR / "tdi_null_zoom_raw.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
