"""Diagnose the diagonal WDM quasi-likelihood on the ungapped 30-day fit.

Orthogonality of the transform does not diagonalise a general non-stationary
covariance.  Under the fitted diagonal model, however,
``z_nm = w_nm / sqrt(S_nm)`` should have negligible residual correlation.  This
script retains the original pooled one-dimensional ACF and also measures the
regime-pooled local correlation over a 3x17 WDM neighbourhood.  The latter is
stratified by location relative to the drifting null comb and by WDM parity.

The local diagnostic uses the cached WDM coefficients when available.  The
saved PSD is in physical units, so omitting the frequency-only WDM calibration
rescales each channel by a positive constant; Pearson correlations are invariant
to that rescaling.

Run after fit_aet_fullband.py:
    python studies/ollie_tdi/z_autocorrelation.py
"""

from __future__ import annotations

import json

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
MAX_TIME_LAG = 1
CORE_WIDTH_HZ = 0.3e-3
FLANK_WIDTH_HZ = 2.0e-3
CACHE = RESULTS_DIR / "knot_map_benchmark" / "wdm_coeffs.npz"


def acf(z: np.ndarray, axis: int, max_lag: int) -> np.ndarray:
    """Pooled autocorrelation of z along ``axis`` at lags 1..max_lag."""
    z = np.moveaxis(z, axis, 0)
    out = []
    for k in range(1, max_lag + 1):
        a, b = z[:-k].ravel(), z[k:].ravel()
        out.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(out)


def _shifted_views(
    array: np.ndarray, dn: int, dm: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned views ``array[n,m]`` and ``array[n+dn,m+dm]``."""
    n0 = slice(max(0, -dn), array.shape[0] - max(0, dn))
    n1 = slice(max(0, dn), array.shape[0] - max(0, -dn))
    m0 = slice(max(0, -dm), array.shape[1] - max(0, dm))
    m1 = slice(max(0, dm), array.shape[1] - max(0, -dm))
    return array[n0, m0], array[n1, m1]


def local_correlation(
    z: np.ndarray,
    regime: np.ndarray,
    *,
    parity: int | None,
    max_dn: int = MAX_TIME_LAG,
    max_dm: int = MAX_LAG,
) -> tuple[np.ndarray, np.ndarray]:
    """Regime-pooled correlation for each local WDM pixel offset.

    Both endpoints must lie in ``regime``.  When ``parity`` is 0 or 1, the
    anchor pixel is restricted by ``(n+m) % 2``; this retains correlations to
    either parity at the shifted endpoint while exposing parity-dependent signs.
    """
    parity_grid = np.indices(z.shape).sum(axis=0) % 2
    shape = (2 * max_dn + 1, 2 * max_dm + 1)
    corr = np.full(shape, np.nan)
    counts = np.zeros(shape, dtype=int)
    for ii, dn in enumerate(range(-max_dn, max_dn + 1)):
        for jj, dm in enumerate(range(-max_dm, max_dm + 1)):
            a, b = _shifted_views(z, dn, dm)
            ma, mb = _shifted_views(regime, dn, dm)
            keep = ma & mb
            if parity is not None:
                pa, _ = _shifted_views(parity_grid, dn, dm)
                keep &= pa == parity
            counts[ii, jj] = int(keep.sum())
            if counts[ii, jj] < 3:
                continue
            if dn == 0 and dm == 0:
                corr[ii, jj] = 1.0
            else:
                corr[ii, jj] = np.corrcoef(a[keep], b[keep])[0, 1]
    return corr, counts


def regime_masks(
    fit: np.lib.npyio.NpzFile, time_days: np.ndarray, freq: np.ndarray
) -> dict[str, np.ndarray]:
    """Classify pixels by distance to the drifting 0.03-Hz null comb."""
    track_006 = np.asarray(fit["null_track_006"])[1]
    if track_006.shape != time_days.shape:
        raise ValueError("0.06-Hz null track does not match the WDM time grid")
    scale = track_006 / np.median(track_006)
    nominal = np.arange(0.03, freq.max() + 0.015, 0.03)
    centers = scale[:, None, None] * nominal[None, :, None]
    distance = np.min(np.abs(freq[None, None, :] - centers), axis=1)
    speed = np.abs(np.gradient(track_006, time_days))
    fast = speed >= np.quantile(speed, 0.75)
    return {
        "smooth": distance > FLANK_WIDTH_HZ,
        "null_flanks": (distance > CORE_WIDTH_HZ) & (distance <= FLANK_WIDTH_HZ),
        "null_cores": distance <= CORE_WIDTH_HZ,
        "rapid_drift": (distance <= FLANK_WIDTH_HZ) & fast[:, None],
    }


def plot_local_correlations(
    z: np.ndarray,
    masks: dict[str, np.ndarray],
    out: "Path",
) -> dict[str, dict[str, object]]:
    """Write the regime x anchor-parity 2D local-correlation figure."""
    parity_cases = {"all": None, "even": 0, "odd": 1}
    results: dict[str, dict[str, object]] = {}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for regime_name, mask in masks.items():
        results[regime_name] = {}
        for parity_name, parity in parity_cases.items():
            corr, counts = local_correlation(z, mask, parity=parity)
            display = corr.copy()
            display[MAX_TIME_LAG, MAX_LAG] = np.nan
            finite = np.abs(display[np.isfinite(display)])
            matrices[(regime_name, parity_name)] = display
            results[regime_name][parity_name] = {
                "max_abs_local_correlation": float(finite.max()) if finite.size else None,
                "min_pair_count": int(counts.min()),
                "max_pair_count": int(counts.max()),
                "correlation": corr.tolist(),
                "pair_counts": counts.tolist(),
            }

    vmax = max(
        np.nanmax(np.abs(matrix)) for matrix in matrices.values()
        if np.any(np.isfinite(matrix))
    )
    vmax = max(vmax, 0.01)
    fig, axes = plt.subplots(
        len(masks), len(parity_cases), figsize=(8.2, 7.4),
        constrained_layout=True, sharex=True, sharey=True,
    )
    image = None
    for row, regime_name in enumerate(masks):
        for col, parity_name in enumerate(parity_cases):
            ax = axes[row, col]
            image = ax.imshow(
                matrices[(regime_name, parity_name)],
                origin="lower", aspect="auto", cmap="coolwarm",
                vmin=-vmax, vmax=vmax,
                extent=(-MAX_LAG - 0.5, MAX_LAG + 0.5,
                        -MAX_TIME_LAG - 0.5, MAX_TIME_LAG + 0.5),
            )
            if row == 0:
                ax.set_title(f"{parity_name} anchors")
            if col == 0:
                ax.set_ylabel(f"{regime_name.replace('_', ' ')}\n$\\Delta n$")
            if row == len(masks) - 1:
                ax.set_xlabel(r"$\Delta m$")
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.set_yticks([-1, 0, 1])
    assert image is not None
    fig.colorbar(image, ax=axes, label=r"corr$(z_{n,m},z_{n+\Delta n,m+\Delta m})$")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results


def main() -> None:
    fit = np.load(RESULTS_DIR / "aet_fullband_A_full.npz")
    if CACHE.exists():
        cached = np.load(CACHE)
        coeffs = np.asarray(cached["coeffs"])
        if not np.array_equal(cached["freq_grid"], fit["freq_grid"]):
            raise ValueError("cached and fitted WDM frequency grids differ")
        # Physical PSD units differ from coefficient units only by a positive
        # frequency-dependent calibration, irrelevant for Pearson correlation.
        z = coeffs / np.sqrt(fit["psd_mean"])
    else:
        aet, dt_raw = load_aet("full")
        data = fft_decimate(aet["A"], DECIMATE)
        dt = dt_raw * DECIMATE
        nt, trim_low = GRID["full"]
        config = PSplineConfig(
            n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
            trim_low_freq_channels=trim_low, trim_time_bins=TRIM_TIME_BINS,
        )
        cal = wdm_white_noise_calibration(data.size, dt, nt, config)
        coeffs, time_grid, freq_grid = wdm_analysis_coefficients(data, dt, nt, config)
        if not np.array_equal(freq_grid, fit["freq_grid"]):
            raise ValueError("recomputed and fitted WDM frequency grids differ")
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

    masks = regime_masks(fit, fit["time_grid_days"], fit["freq_grid"])
    local_out = RESULTS_DIR / "tdi_z_local_correlation.png"
    local = plot_local_correlations(z, masks, local_out)
    summary = {
        "definition": {
            "max_time_lag": MAX_TIME_LAG,
            "max_frequency_lag": MAX_LAG,
            "core_width_hz": CORE_WIDTH_HZ,
            "flank_width_hz": FLANK_WIDTH_HZ,
            "rapid_drift_quantile": 0.75,
            "parity": "anchor pixel (n+m) mod 2",
        },
        "regime_pixel_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "results": local,
    }
    json_out = RESULTS_DIR / "tdi_z_local_correlation.json"
    json_out.write_text(json.dumps(summary, indent=2))
    for regime_name, by_parity in local.items():
        values = " ".join(
            f"{name}={metrics['max_abs_local_correlation']:.4f}"
            for name, metrics in by_parity.items()
            if metrics["max_abs_local_correlation"] is not None
        )
        print(f"[local] {regime_name}: max |r| {values}")
    print(f"[out] {local_out}")
    print(f"[out] {json_out}")


if __name__ == "__main__":
    main()
