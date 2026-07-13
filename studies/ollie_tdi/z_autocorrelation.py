"""Measure the WDM independence approximation: autocorrelation of whitened z.

The WDM likelihood treats the coefficients as independent, which is exact only
for a perfectly orthogonal transform. This post-processes the ungapped 30-day
fit (no refit): if the fitted surface is correct and the transform orthogonal,
z_nm = w_nm / sqrt(S_nm) is i.i.d. N(0,1), so any residual autocorrelation in
time (across WDM bins at fixed channel) or frequency (across channels at fixed
bin) measures the approximation error directly. Appendix figure for the paper.

Run after fit_aet_fullband.py:
    python studies/ollie_tdi/z_autocorrelation.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from fit_aet_fullband import (
    DECIMATE,
    GRID,
    N_KNOTS_LIN,
    N_KNOTS_LOG,
    RESULTS_DIR,
    TRIM_TIME_BINS,
    fft_decimate,
    load_aet,
)

from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients
from tv_pspline_psd.datasets import wdm_white_noise_calibration

set_paper_style()

MAX_LAG = 8


def acf(z: np.ndarray, axis: int, max_lag: int) -> np.ndarray:
    """Pooled autocorrelation of z along ``axis`` at lags 1..max_lag."""
    z = np.moveaxis(z, axis, 0)
    out = []
    for k in range(1, max_lag + 1):
        a, b = z[:-k].ravel(), z[k:].ravel()
        out.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(out)


def main() -> None:
    fit = np.load(RESULTS_DIR / "aet_fullband_A_full.npz")
    aet, dt_raw = load_aet("full")
    data = fft_decimate(aet["A"], DECIMATE)
    dt = dt_raw * DECIMATE
    nt, trim_low = GRID["full"]
    config = PSplineConfig(
        n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
        trim_low_freq_channels=trim_low, trim_time_bins=TRIM_TIME_BINS,
    )
    cal = wdm_white_noise_calibration(data.size, dt, nt, config)
    coeffs, _, _ = wdm_analysis_coefficients(data, dt, nt, config)
    z = coeffs / np.sqrt(fit["psd_mean"] * cal[None, :] / (2.0 * dt))

    lags = np.arange(1, MAX_LAG + 1)
    r_t, r_f = acf(z, 0, MAX_LAG), acf(z, 1, MAX_LAG)
    # z is linear in the coefficients; |z| probes power-level dependence too.
    ra_t, ra_f = acf(np.abs(z), 0, MAX_LAG), acf(np.abs(z), 1, MAX_LAG)
    # Studentizing per channel removes the fixed per-channel miscalibration
    # footprint of the smoothed null cores (std z != 1 there), isolating any
    # genuine coefficient dependence left by the transform.
    zs = z / z.std(axis=0, keepdims=True)
    rs_t, rs_f = acf(np.abs(zs), 0, MAX_LAG), acf(np.abs(zs), 1, MAX_LAG)
    floor = 1 / np.sqrt(z.size)
    print(f"n = {z.size}, 1/sqrt(n) floor = {floor:.1e}")
    print(f"time  lag-1: r(z) = {r_t[0]:+.4f}, r(|z|) = {ra_t[0]:+.4f}, "
          f"studentized r(|z|) = {rs_t[0]:+.4f}")
    print(f"freq  lag-1: r(z) = {r_f[0]:+.4f}, r(|z|) = {ra_f[0]:+.4f}, "
          f"studentized r(|z|) = {rs_f[0]:+.4f}")
    print(f"max |r| over lags 1-{MAX_LAG}: "
          f"time {np.abs(r_t).max():.4f}, freq {np.abs(r_f).max():.4f}")

    # Single panel, z only: the |z| / studentised-|z| detail is quoted in the
    # text (it is a fixed null-core footprint, not lag structure).
    fig, ax = plt.subplots(figsize=(3.4, 2.2), constrained_layout=True)
    ax.axhspan(-3 * floor, 3 * floor, color="0.92", lw=0,
               label=r"$\pm 3/\sqrt{n}$")
    ax.axhline(0, color="black", lw=0.7)
    off = 0.12
    for r, x0, color, mk, label in ((r_t, -off, "tab:blue", "o", "time-bin lag"),
                                    (r_f, +off, "tab:orange", "s",
                                     "frequency-channel lag")):
        ml, sl, bl = ax.stem(lags + x0, r, basefmt=" ", label=label)
        plt.setp(sl, color=color, lw=1.2)
        plt.setp(ml, color=color, marker=mk, ms=4)
    ax.set_xticks(lags)
    ax.set_ylim(-0.01, 0.01)
    ax.set_xlabel("lag")
    ax.set_ylabel(r"autocorrelation of $z$")
    ax.legend(fontsize=7)

    out = RESULTS_DIR / "tdi_z_autocorrelation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
