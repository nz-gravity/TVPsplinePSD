"""Diagnose the oscillations in manuscript Figure 7.

This script consumes non-destructive sensitivity fits produced by
``fit_aet_fullband.py --tag-suffix fig7_ktXX``. It compares time-knot counts,
three null-location functionals, per-chain tracks, whitening, and two arm-length
proxies. It writes a diagnostic figure and JSON summary without modifying the
production Figure 7 artifacts.

Run after the OzStar ``fig7_sensitivity.sh`` array completes::

    python studies/ollie_tdi/fig7_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis

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
    load_aet,
)
from tv_pspline_psd import PSplineConfig, set_paper_style, wdm_analysis_coefficients
from tv_pspline_psd.datasets import wdm_white_noise_calibration

TIME_KNOTS = (8, 12, 16, 20)
EXTRACTORS = ("centroid", "quadratic", "spline_min")
COLORS = dict(zip(EXTRACTORS, ("tab:blue", "tab:orange", "tab:green")))


def _fit_path(time_knots: int, gapped: bool) -> Path:
    gap = "_gaps" if gapped else ""
    return RESULTS_DIR / f"aet_fullband_A_full{gap}_fig7_kt{time_knots:02d}.npz"


def _diag_path(time_knots: int, gapped: bool) -> Path:
    return _fit_path(time_knots, gapped).with_name(
        _fit_path(time_knots, gapped).stem + "_diag.json"
    )


def _prediction(track: np.ndarray, time_days: np.ndarray, time_ltt: np.ndarray,
                light_time: np.ndarray) -> np.ndarray:
    lam = np.interp(time_days, time_ltt, light_time / light_time[0])
    return track.mean() * lam.mean() / lam


def _track_metrics(track: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = 1e6 * (track - prediction)
    return {
        "correlation": float(np.corrcoef(track, prediction)[0, 1]),
        "residual_rms_uHz": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs_uHz": float(np.max(np.abs(residual))),
        "drift_range_uHz": float(1e6 * np.ptp(track)),
    }


def _load_light_times() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    with h5py.File(DATA_FULL) as h:
        arms = {key: h[f"ltts/ltt_{key}"][:] for key in
                ("12", "13", "21", "23", "31", "32")}
        theory = h["noise_estimates/AET"][:]
    all_arm = np.mean(list(arms.values()), axis=0)
    # A=(Z-X)/sqrt(2): average the four arms entering X and the four entering Z.
    # This is still a scalar-shift proxy, but it respects the channel composition
    # and weights the shared 13/31 arms twice instead of averaging all six equally.
    x_arm = np.mean([arms[k] for k in ("12", "21", "13", "31")], axis=0)
    z_arm = np.mean([arms[k] for k in ("31", "13", "32", "23")], axis=0)
    a_arm = 0.5 * (x_arm + z_arm)
    return all_arm, a_arm, theory, bool(np.any(np.abs(theory) > 0.0))


def _whitening_inputs() -> tuple[dict[bool, np.ndarray], np.ndarray, float]:
    aet, dt_raw = load_aet("full")
    data = fft_decimate(aet["A"], DECIMATE)
    dt = dt_raw * DECIMATE
    nt, trim_low = GRID["full"]
    config = PSplineConfig(
        n_interior_knots_freq=N_KNOTS_LOG + N_KNOTS_LIN,
        trim_low_freq_channels=trim_low,
        trim_time_bins=TRIM_TIME_BINS,
    )
    clean, time_grid, _ = wdm_analysis_coefficients(data, dt, nt, config)
    example = np.load(_fit_path(TIME_KNOTS[0], True))
    gaps = [tuple(row) for row in example["gaps_s"]]
    gated, gated_time, _ = wdm_analysis_coefficients(
        gate_gaps(data, dt, gaps), dt, nt, config
    )
    keep = good_time_bins(gated_time, data.size * dt, gaps, nt)
    cal = wdm_white_noise_calibration(data.size, dt, nt, config)
    return {False: clean, True: gated[keep]}, cal, dt


def main() -> None:
    set_paper_style()
    missing = [str(_fit_path(k, g)) for k in TIME_KNOTS for g in (False, True)
               if not _fit_path(k, g).exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 7 sensitivity fits:\n" + "\n".join(missing))

    fits = {(k, g): np.load(_fit_path(k, g)) for k in TIME_KNOTS for g in (False, True)}
    coeffs, cal, dt = _whitening_inputs()
    all_arm, a_arm, theory, theory_available = _load_light_times()
    longest_time = max(float(f["time_grid_days"][-1] + f["time_grid_days"][0])
                       for f in fits.values())
    time_ltt = np.linspace(0.0, longest_time, all_arm.size)

    summary: dict[str, object] = {
        "theoretical_aet_covariance_available": theory_available,
        "theoretical_aet_covariance_shape": list(theory.shape),
        "note": (
            "noise_estimates/AET is zero-valued; exact transfer-function validation "
            "is unavailable from this file."
            if not theory_available else
            "A nonzero theoretical AET covariance is present but is not used here."
        ),
        "fits": {},
    }

    fig, axes = plt.subplots(2, len(TIME_KNOTS), figsize=(11.0, 5.0),
                             sharex=True, constrained_layout=True)
    for col, knots in enumerate(TIME_KNOTS):
        for row, (f0, prefix) in enumerate(((0.06, "null_track_006"),
                                            (0.12, "null_track_012"))):
            ax = axes[row, col]
            for gapped, linestyle in ((False, "-"), (True, "--")):
                fit = fits[(knots, gapped)]
                time_days = fit["time_grid_days"]
                for extractor in EXTRACTORS:
                    track = fit[f"{prefix}_{extractor}"][1]
                    ax.plot(time_days, 1e6 * (track - track.mean()),
                            color=COLORS[extractor], ls=linestyle, lw=0.9,
                            alpha=1.0 if not gapped else 0.65)
                if not gapped:
                    centroid = fit[f"{prefix}_centroid"][1]
                    prediction = _prediction(centroid, time_days, time_ltt, a_arm)
                    ax.plot(time_days, 1e6 * (prediction - prediction.mean()),
                            color="black", lw=0.8)
            ax.set_title(f"{knots} time knots" if row == 0 else "")
            if col == 0:
                ax.set_ylabel(rf"{f0:.2f} Hz track [$\mu$Hz]")
            if row == 1:
                ax.set_xlabel("time [days]")

    for extractor in EXTRACTORS:
        axes[0, -1].plot([], [], color=COLORS[extractor], label=extractor)
    axes[0, -1].plot([], [], color="0.3", ls="--", label="gapped")
    axes[0, -1].plot([], [], color="black", label="A-arm proxy")
    axes[0, -1].legend(fontsize=6, loc="upper left")

    for knots in TIME_KNOTS:
        for gapped in (False, True):
            fit = fits[(knots, gapped)]
            tag = f"kt{knots:02d}_{'gapped' if gapped else 'ungapped'}"
            with open(_diag_path(knots, gapped)) as handle:
                diag = json.load(handle)
            s_coeff = fit["psd_mean"] * cal[None, :] / (2.0 * dt)
            z = coeffs[gapped] / np.sqrt(s_coeff)
            entry = {
                "sampler": diag,
                "whitening": {
                    "mean_z2": float(np.mean(z**2)),
                    "time_z2_rmse": float(np.sqrt(np.mean((np.mean(z**2, axis=1) - 1.0)**2))),
                    "freq_z2_rmse": float(np.sqrt(np.mean((np.mean(z**2, axis=0) - 1.0)**2))),
                    "excess_kurtosis": float(kurtosis(z.ravel())),
                },
                "nulls": {},
            }
            for f0, prefix in ((0.06, "null_track_006"), (0.12, "null_track_012")):
                tracks = {name: fit[f"{prefix}_{name}"][1] for name in EXTRACTORS}
                centroid = tracks["centroid"]
                by_chain = fit[f"{prefix}_centroid_by_chain"][:, 1, :]
                null_entry = {
                    "all_arm_proxy": _track_metrics(
                        centroid, _prediction(centroid, fit["time_grid_days"], time_ltt, all_arm)
                    ),
                    "a_channel_arm_proxy": _track_metrics(
                        centroid, _prediction(centroid, fit["time_grid_days"], time_ltt, a_arm)
                    ),
                    "extractor_rms_uHz": {
                        name: float(1e6 * np.sqrt(np.mean((track - centroid)**2)))
                        for name, track in tracks.items() if name != "centroid"
                    },
                    "chain_median_rms_uHz": float(
                        1e6 * np.sqrt(np.mean((by_chain[0] - by_chain[1])**2))
                    ),
                }
                entry["nulls"][f"{f0:.2f}"] = null_entry
            summary["fits"][tag] = entry

    output_figure = RESULTS_DIR / "fig7_sensitivity.png"
    output_json = RESULTS_DIR / "fig7_sensitivity.json"
    fig.savefig(output_figure, dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(output_json, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[out] {output_figure}")
    print(f"[out] {output_json}")


if __name__ == "__main__":
    main()
