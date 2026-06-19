"""Recover the time-varying A/E/T noise PSD from real LISA-challenge data.

Analyzes the ``noise6a`` dataset (one year of XYZ gen-2 TDI from the
``lisa_datagen`` project: instrument noise + annually modulated Galactic
confusion + one bright GB at 3.21 mHz). We transform to A/E/T, band-limit to the
confusion band, and estimate each channel's time-varying PSD with the whitened
P-spline. The recovered seasonal modulation is validated against a time-resolved
(Welch-per-block) empirical PSD -- the correct reference for what is actually in
each channel.

Set ``NOISE6A_H5`` to the dataset path. Saves ``notes/figures/noise6a_*.png``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from datasets.lisa_noise6a import load_noise6a_aet
from wdm_psd import PSplineConfig, run_wdm_psd_mcmc, save_figure

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
H5 = os.getenv("NOISE6A_H5",
               str(Path.home() / "Documents/projects/lisa_datagen/data/noise6a_tdi.h5"))
# N = NT*NF sets the Nyquist after decimation; keep it large enough to cover the
# confusion band (~3 mHz) plus the bright GB at 3.21 mHz (Nyquist ~3.9 mHz here).
NT, NF = 240, 1020
CONF_BAND = (1.0e-3, 3.0e-3)


def empirical_modulation(series, dt, gb_f0, n_blocks):
    """Confusion-band power per time block (away from the bright GB line)."""
    block = len(series) // n_blocks
    power = np.empty(n_blocks)
    for b in range(n_blocks):
        fr, p = welch(series[b * block:(b + 1) * block], fs=1.0 / dt,
                      nperseg=min(block, 2048))
        m = (fr > CONF_BAND[0]) & (fr < CONF_BAND[1]) & (np.abs(fr - gb_f0) > 1.5e-4)
        power[b] = p[m].mean()
    return power / power.mean()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(H5).exists():
        raise SystemExit(f"noise6a dataset not found at {H5}; set NOISE6A_H5.")
    data = load_noise6a_aet(H5, nt=NT, nf=NF)
    print(f"noise6a A/E/T: N={data['n_total']} dt={data['dt']:.1f}s "
          f"Nyquist={data['nyquist_hz'] * 1e3:.2f} mHz, bright GB at {data['gb_f0'] * 1e3:.2f} mHz")

    cfg = PSplineConfig(n_interior_knots_time=12, n_interior_knots_freq=10,
                        trim_low_freq_channels=2)
    results, emp = {}, {}
    t0 = time.time()
    for ch in "AET":
        res = run_wdm_psd_mcmc(data[ch], dt=data["dt"], nt=NT, config=cfg,
                               n_warmup=250, n_samples=250, random_seed=1,
                               store_log_psd_samples=False)
        results[ch] = res
        emp[ch] = empirical_modulation(data[ch], data["dt"], data["gb_f0"], NT // 2)
        band = ((res["freq_grid"] > CONF_BAND[0]) & (res["freq_grid"] < CONF_BAND[1])
                & (np.abs(res["freq_grid"] - data["gb_f0"]) > 1.5e-4))
        mod = res["psd_mean"][:, band].mean(axis=1)
        mod /= mod.mean()
        emp_on_grid = np.interp(res["time_grid"], (np.arange(NT // 2) + 0.5) / (NT // 2), emp[ch])
        corr = float(np.corrcoef(mod, emp_on_grid)[0, 1])
        print(f"  channel {ch}: div={res['divergences']}  "
              f"modulation depth est={mod.max() / mod.min():.2f} "
              f"emp={emp[ch].max() / emp[ch].min():.2f}  corr(est,emp)={corr:.2f}  "
              f"({time.time() - t0:.0f}s)")

    # Figure 1: recovered A/E/T log-PSD surfaces.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True, sharey=True)
    for ax, ch in zip(axes, "AET"):
        r = results[ch]
        mesh = ax.pcolormesh(r["time_grid"], r["freq_grid"] * 1e3,
                             np.log(r["psd_mean"]).T, shading="nearest", cmap="viridis")
        ax.axhline(data["gb_f0"] * 1e3, color="white", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(f"Channel {ch}: recovered $\\log S(t,f)$")
        ax.set_xlabel("Rescaled time $u$")
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("Frequency [mHz]")
    save_figure(fig, FIG_DIR / "noise6a_surfaces.png")

    # Figure 2: recovered vs empirical seasonal modulation per channel.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True, sharey=True)
    for ax, ch in zip(axes, "AET"):
        r = results[ch]
        band = ((r["freq_grid"] > CONF_BAND[0]) & (r["freq_grid"] < CONF_BAND[1])
                & (np.abs(r["freq_grid"] - data["gb_f0"]) > 1.5e-4))
        mod = r["psd_mean"][:, band].mean(axis=1)
        mod /= mod.mean()
        u_emp = (np.arange(NT // 2) + 0.5) / (NT // 2)
        ax.plot(u_emp, emp[ch], color="black", lw=1.2, alpha=0.6, label="empirical (Welch)")
        ax.plot(r["time_grid"], mod, color="tab:blue", lw=2.0, label="P-spline estimate")
        ax.set_title(f"Channel {ch} confusion-band modulation")
        ax.set_xlabel("Rescaled time $u$")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Normalized band power")
    save_figure(fig, FIG_DIR / "noise6a_modulation.png")
    print(f"Saved noise6a figures to {FIG_DIR}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
