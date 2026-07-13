"""TV-PSD fit to externally simulated LISA TDI noise (datasets/ollie_data).

The data are 30 days of second-generation TDI from a full
lisainstrument+pytdi pipeline (A2/E2 channels, fs = 4 Hz, fractional
frequency). The series is brick-wall decimated to the mHz band and fitted
with the WDM log-P-spline estimator. The file's own daily spectral
estimates (``noise_estimates/AET``) provide an external reference surface.

The noise is stationary by construction (no Galactic confusion), so this
doubles as a null test: the time-varying fit should recover a
time-constant surface that matches the pipeline's daily estimates.

Run:
    python studies/ollie_tdi/fit_ollie_tdi.py            # A channel
    python studies/ollie_tdi/fit_ollie_tdi.py --channel E2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from tv_pspline_psd import (
    PSplineConfig,
    run_wdm_psd_mcmc,
    set_paper_style,
    summarize_mcmc_diagnostics,
)
from tv_pspline_psd.datasets import wdm_white_noise_calibration

set_paper_style()

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "ollie_data" / "simulated_noise_30_days_L1_ext.h5"
RESULTS_DIR = REPO / "studies" / "results" / "ollie_tdi"

DECIMATE = 648          # 0.25 s -> 162 s, Nyquist 3.09 mHz (LISA mHz band)
NT = 50                 # WDM time bins (~0.6 day resolution over 30 days)
CHANNEL_INDEX = {"A2": 0, "E2": 1, "T2": 2}


def load_channel(channel: str) -> tuple[np.ndarray, float]:
    with h5py.File(DATA) as h:
        x = h[f"tdis/{channel}"][:]
        dt = 0.25
    return x, dt


def fft_decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Brick-wall lowpass + downsample by cropping the rFFT."""
    n_new = x.size // q
    spec = np.fft.rfft(x[: n_new * q])
    return np.fft.irfft(spec[: n_new // 2 + 1], n=n_new) / q


# ponytail: the file's noise_estimates/AET arrays are all zero (placeholders),
# so the external reference is daily Welch estimates from the raw 4 Hz series.
def daily_welch_reference(raw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-day one-sided Welch PSDs from the full-rate series (n_days, n_freq)."""
    nday = int(86400 * fs)
    days = raw[: (raw.size // nday) * nday].reshape(-1, nday)
    freqs, psd = welch(days, fs=fs, nperseg=2**16, axis=-1, detrend="linear")
    return freqs, psd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="A2", choices=list(CHANNEL_INDEX))
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=2)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw, dt_raw = load_channel(args.channel)
    data = fft_decimate(raw, DECIMATE)
    dt = dt_raw * DECIMATE
    n_total = data.size
    print(f"[data] {args.channel}: {raw.size} -> {n_total} samples, dt={dt:.0f}s, "
          f"Nyquist={1/(2*dt)*1e3:.2f} mHz, T={n_total*dt/86400:.1f} days")

    # Trim 2 edge time bins (>1 day): the brick-wall decimated series is not
    # periodic, so the WDM boundary bins carry wrap-around power.
    config = PSplineConfig(trim_low_freq_channels=2, trim_time_bins=2)
    t0 = time.perf_counter()
    res = run_wdm_psd_mcmc(
        data, dt=dt, nt=NT, config=config,
        n_warmup=args.n_warmup, n_samples=args.n_samples,
        num_chains=args.num_chains, random_seed=0,
    )
    total_s = time.perf_counter() - t0
    diag = summarize_mcmc_diagnostics(res)
    print(f"[fit] wall={total_s:.0f}s sampling={res['nuts_runtime_s']:.0f}s "
          f"div={diag['divergences']} "
          f"rhat(phi)<= {max(diag['phi_time']['r_hat'], diag['phi_freq']['r_hat']):.3f}")

    # Convert WDM-coefficient power to a one-sided PSD in 1/Hz:
    # E[w^2] = C_m * S_dig with S_dig = S_onesided / (2 dt).
    cal = wdm_white_noise_calibration(n_total, dt, NT, config)
    to_psd = 2.0 * dt / cal[None, :]
    S_est = res["psd_mean"] * to_psd
    S_lo = res["psd_lower"] * to_psd
    S_hi = res["psd_upper"] * to_psd
    tg_days = res["time_grid"] * n_total * dt / 86400
    fg = res["freq_grid"]

    ref_f, ref_daily = daily_welch_reference(raw, 1 / dt_raw)
    f_w, P_w = welch(data, fs=1 / dt, nperseg=n_total // 8)

    np.savez(
        RESULTS_DIR / f"ollie_fit_{args.channel}.npz",
        time_grid_days=tg_days, freq_grid=fg,
        psd_mean=S_est, psd_lower=S_lo, psd_upper=S_hi,
        welch_f=f_w, welch_psd=P_w,
        daily_welch_freqs=ref_f, daily_welch_psd=ref_daily,
        runtime_s=total_s, nuts_runtime_s=res["nuts_runtime_s"],
        divergences=diag["divergences"],
    )
    with open(RESULTS_DIR / f"ollie_fit_{args.channel}_diag.json", "w") as fp:
        json.dump(diag, fp, indent=2, default=float)

    # --- Figures ---
    # 1) Recovered surface + raw WDM log-power.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True, sharey=True)
    raw_pow = np.log(res["power"] + 1e-30) + np.log(to_psd)
    mesh0 = axes[0].pcolormesh(tg_days, fg * 1e3, raw_pow.T, shading="auto", cmap="viridis")
    axes[0].set_title("raw WDM log power")
    mesh1 = axes[1].pcolormesh(tg_days, fg * 1e3, np.log(S_est).T, shading="auto",
                               cmap="viridis", vmin=mesh0.get_clim()[0], vmax=mesh0.get_clim()[1])
    axes[1].set_title(r"posterior mean $\log \hat S(t,f)$")
    for ax in axes:
        ax.set_xlabel("time [days]")
    axes[0].set_ylabel("f [mHz]")
    fig.colorbar(mesh1, ax=axes, shrink=0.85, label=r"$\log S$ [1/Hz]")
    fig.savefig(RESULTS_DIR / f"ollie_surface_{args.channel}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Time-averaged spectrum vs Welch and the pipeline's daily estimates.
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    # Below ~0.3 mHz the 1-day segments are leakage-dominated (red noise below
    # the daily resolution), so show the daily curves only above that.
    band = (ref_f >= max(fg.min(), 3e-4)) & (ref_f <= fg.max())
    ax.loglog(ref_f[band] * 1e3, ref_daily[:, band].T, color="0.8", lw=0.4,
              label=None)
    ax.plot([], [], color="0.8", lw=0.8, label="daily Welch")
    ax.loglog(f_w[1:] * 1e3, P_w[1:], color="0.4", lw=0.8, label="Welch (30 d)")
    ax.loglog(fg * 1e3, S_est.mean(axis=0), color="tab:blue", label="TV fit (time avg)")
    ax.fill_between(fg * 1e3, S_lo.mean(axis=0), S_hi.mean(axis=0),
                    color="tab:blue", alpha=0.3)
    ax.set_xlabel("f [mHz]")
    ax.set_ylabel(r"$S(f)$ [1/Hz]")
    ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / f"ollie_spectrum_{args.channel}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Stationarity check: fractional variation of S(t) at fixed frequencies.
    fig, ax = plt.subplots(figsize=(3.6, 2.4), constrained_layout=True)
    for f_target in (0.5e-3, 1e-3, 2e-3):
        j = int(np.argmin(np.abs(fg - f_target)))
        med = np.median(S_est[:, j])
        ax.plot(tg_days, S_est[:, j] / med, label=f"{fg[j]*1e3:.2f} mHz")
        ax.fill_between(tg_days, S_lo[:, j] / med, S_hi[:, j] / med, alpha=0.2)
    ax.axhline(1.0, color="black", ls=":")
    ax.set_xlabel("time [days]")
    ax.set_ylabel(r"$\hat S(t,f)/\mathrm{med}_t\,\hat S$")
    ax.legend(fontsize=7)
    fig.savefig(RESULTS_DIR / f"ollie_stationarity_{args.channel}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[out] figures + npz in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
