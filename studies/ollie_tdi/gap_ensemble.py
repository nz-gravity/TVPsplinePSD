"""Gap-robustness ensemble: refit the 30-day A channel under many gap schedules.

The paper's gapped fit uses a single LISA-like schedule (seed 1). This repeats
the fit for --n-seeds independent unscheduled-outage realisations of the same
data and scores each on (a) whitening calibration (pooled mean z^2, std z, and
excess kurtosis with the +-2 mHz null comb excluded) and (b) null tracking
(correlation of the fitted 0.06 / 0.12 Hz null trajectories with the
parameter-free 1/Lbar(t) armlength prediction). One summary npz + printout;
supports the "gaps cost precision, not calibration" claim with an ensemble
instead of an anecdote.

Run after downloading the 30-day data (background job, ~4 min/seed):
    python studies/ollie_tdi/gap_ensemble.py --n-seeds 10
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
from fit_aet_fullband import (
    DATA_FULL,
    DECIMATE,
    GRID,
    N_KNOTS_LIN,
    N_KNOTS_LOG,
    RESULTS_DIR,
    TRIM_TIME_BINS,
    fft_decimate,
    gate_gaps,
    good_time_bins,
    lisa_like_gaps,
    load_aet,
    warp_freq,
)
from gap_compare import null_track
from scipy.stats import kurtosis

from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    summarize_mcmc_diagnostics,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.datasets import wdm_white_noise_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-warmup", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    aet, dt_raw = load_aet("full")
    clean = fft_decimate(aet["A"], DECIMATE)
    dt = dt_raw * DECIMATE
    t_obs_s = clean.size * dt
    nt, trim_low = GRID["full"]
    config = PSplineConfig(
        n_interior_knots_time=16,  # match fit_aet_fullband's --time-knots
        n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
        trim_low_freq_channels=trim_low, trim_time_bins=TRIM_TIME_BINS,
        centered=True,
    )
    cal = wdm_white_noise_calibration(clean.size, dt, nt, config)

    with h5py.File(DATA_FULL) as h:
        ltts = np.stack([h[f"ltts/ltt_{k}"][:] for k in
                         ("12", "13", "21", "23", "31", "32")])
    L_bar = ltts.mean(axis=0)

    rows = []
    for seed in range(1, args.n_seeds + 1):
        gaps = lisa_like_gaps(t_obs_s, seed=seed)
        data = gate_gaps(clean, dt, gaps)
        coeffs, time_grid, freq_grid = wdm_analysis_coefficients(
            data, dt, nt, config)
        keep = good_time_bins(time_grid, t_obs_s, gaps, nt)
        coeffs, time_grid = coeffs[keep], time_grid[keep]

        res = fit_log_pspline_surface(
            coeffs[None, :, :], time_grid, warp_freq(freq_grid), config=config,
            n_warmup=args.n_warmup, n_samples=args.n_samples,
            num_chains=2, random_seed=seed,
        )
        diag = summarize_mcmc_diagnostics(res)

        z = coeffs / np.sqrt(res["psd_mean"])
        nulls = np.arange(0.03, freq_grid.max(), 0.03)
        dist = np.min(np.abs(freq_grid[:, None] - nulls[None, :]), axis=1)
        kurt = kurtosis(z[:, dist > 2e-3].ravel())

        tg_days = time_grid * t_obs_s / 86400
        t_L = np.linspace(0, t_obs_s / 86400, L_bar.size)
        lam = np.interp(tg_days, t_L, L_bar / L_bar[0])
        corr = {}
        for f0 in (0.06, 0.12):
            fnull = null_track(tg_days, freq_grid, res["psd_mean"] * cal, f0)
            corr[f0] = np.corrcoef(fnull, fnull.mean() * lam.mean() / lam)[0, 1]

        row = dict(
            seed=seed, n_gaps=len(gaps),
            gated_h=sum(t1 - t0 for t0, t1 in gaps) / 3600,
            dropped_bins=int(np.count_nonzero(~keep)),
            mean_z2=float(np.mean(z**2)), std_z=float(np.std(z)),
            kurt_nonull=float(kurt),
            corr_006=float(corr[0.06]), corr_012=float(corr[0.12]),
            divergences=int(diag["divergences"]),
        )
        rows.append(row)
        print("[seed {seed}] gaps={n_gaps} gated={gated_h:.0f}h "
              "dropped={dropped_bins} mean_z2={mean_z2:.4f} "
              "kurt={kurt_nonull:+.3f} corr06={corr_006:+.3f} "
              "corr12={corr_012:+.3f} div={divergences}".format(**row))

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(RESULTS_DIR / "gap_ensemble.npz", **out)
    print(f"\n=== gap ensemble ({len(rows)} schedules) ===")
    print(f"gated hours     : {out['gated_h'].min():.0f}..{out['gated_h'].max():.0f}")
    print(f"dropped bins    : {out['dropped_bins'].min()}..{out['dropped_bins'].max()} of 120")
    print(f"pooled mean z^2 : {out['mean_z2'].mean():.4f} +- {out['mean_z2'].std():.4f}")
    print(f"kurt (no null)  : {out['kurt_nonull'].mean():+.3f} +- {out['kurt_nonull'].std():.3f}")
    print(f"corr @ 0.06 Hz  : min {out['corr_006'].min():+.3f}")
    print(f"corr @ 0.12 Hz  : min {out['corr_012'].min():+.3f}")
    print(f"divergences     : {int(out['divergences'].sum())}")


if __name__ == "__main__":
    main()
