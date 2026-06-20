"""Two-likelihood LS2 simulation study figures (manuscript Section 4).

Both observation models -- the WDM coefficient likelihood and the Tang zigzag
moving-periodogram dynamic Whittle -- are fitted with the *same* whitened
P-spline model, so the comparison isolates the time-frequency representation.

Produces:
  * ``sim_three_panel.png``  (Fig 1) -- true log-PSD, WDM posterior median, and
    moving-periodogram posterior median for a single LS2 realization, shared
    colour scale.
  * ``sim_mse_coverage.png`` (Fig 2) -- MSE_{log f} (upper) and 90% credible-
    interval coverage (lower) versus the number of observations, one curve per
    likelihood, each point averaged over ``--repeats`` realizations.

    python notes/scripts/make_sim_study_figures.py --repeats 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from datasets import simulate_ls2, true_psd_ls2, wdm_white_noise_calibration
from tv_pspline_psd import (
    PSplineConfig,
    interval_coverage,
    run_tang_dynamic_whittle_mcmc,
    run_wdm_psd_mcmc,
    set_paper_style,
    tang_moving_periodogram,
)

set_paper_style()

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT = 0.1
NF = 24
NT_VALUES = (24, 48, 96, 192)
TANG_M, TANG_THIN = 16, 2
U_COMMON = np.linspace(0.05, 0.95, 60)
F_COMMON = np.linspace(0.6, 4.4, 60)

WDM_CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=10)
TANG_CONFIG = PSplineConfig(n_interior_knots_time=8, n_interior_knots_freq=6)


def _tang_calibration() -> float:
    return float(np.mean(tang_moving_periodogram(
        np.random.default_rng(1).standard_normal(8192), m=TANG_M, thin=TANG_THIN)["mi"]))


def _to_common(time_grid, freq_grid, log_field):
    interp = RegularGridInterpolator(
        (time_grid, freq_grid), log_field, bounds_error=False, fill_value=None)
    uu, ff = np.meshgrid(U_COMMON, F_COMMON, indexing="ij")
    return interp(np.stack([uu.ravel(), ff.ravel()], axis=-1)).reshape(uu.shape)


def _fit_both(data, nt, seed, cal_wdm, cal_tang):
    rw = run_wdm_psd_mcmc(data, dt=DT, nt=nt, config=WDM_CONFIG,
                          n_warmup=250, n_samples=250, random_seed=seed)
    rt = run_tang_dynamic_whittle_mcmc(data, dt=DT, m=TANG_M, thin=TANG_THIN,
                                       config=TANG_CONFIG, n_warmup=250,
                                       n_samples=250, random_seed=seed)
    return rw, rt


def _metrics(res, cal, log_f0_common):
    cal = np.asarray(cal)
    cal_b = cal if cal.ndim == 0 else cal[None, :]  # scalar (Tang) or per-channel (WDM)
    # MSE on the common grid (rescaled to the analytic PSD scale).
    fitted = _to_common(res["time_grid"], res["freq_grid"],
                        np.log(res["psd_mean"] / cal_b + 1e-12))
    mse = float(np.mean((fitted - log_f0_common) ** 2))
    # Coverage on the estimator's native grid against the calibrated truth.
    true_native = cal_b * true_psd_ls2(res["time_grid"], res["freq_grid"], DT)
    cov = interval_coverage(true_native, res["psd_lower"], res["psd_upper"])
    # Width of the 90% credible interval on log S (scale-free, comparable across
    # representations), averaged over the grid.
    ci_width = float(np.mean(np.log(res["psd_upper"] + 1e-30)
                             - np.log(res["psd_lower"] + 1e-30)))
    return mse, cov, ci_width


def _render_metrics(durations: np.ndarray, raw: dict[str, list]) -> None:
    """Render the MSE / coverage / CI-width vs. n figure from per-N samples."""
    med = lambda key: np.array([np.median(a) for a in raw[key]])
    q = lambda key, p: np.array([np.percentile(a, p) for a in raw[key]])

    def _band(ax, key, color, marker, label):
        ax.plot(durations, med(key), marker, color=color, label=label)
        ax.fill_between(durations, q(key, 25), q(key, 75), color=color, alpha=0.18)

    fig, (ax_m, ax_c, ax_w) = plt.subplots(3, 1, figsize=(6.5, 9.0), sharex=True,
                                           constrained_layout=True)
    _band(ax_m, "wm", "tab:blue", "o-", "WDM")
    _band(ax_m, "tm", "tab:orange", "s--", "Moving periodogram")
    ax_m.set_xscale("log"); ax_m.set_yscale("log")
    ax_m.set_ylabel(r"$\mathrm{MSE}_{\log f}$ (median, IQR)")
    ax_m.legend()

    ax_c.semilogx(durations, np.array([np.mean(a) for a in raw["wc"]]), "o-", color="tab:blue")
    ax_c.semilogx(durations, np.array([np.mean(a) for a in raw["tc"]]), "s--", color="tab:orange")
    ax_c.axhline(0.9, ls=":", color="black", label="nominal 90%")
    ax_c.set_ylim(0.0, 1.0); ax_c.set_ylabel("90% coverage"); ax_c.legend()

    _band(ax_w, "ww", "tab:blue", "o-", "WDM")
    _band(ax_w, "tw", "tab:orange", "s--", "Moving periodogram")
    ax_w.set_xscale("log"); ax_w.set_yscale("log")
    ax_w.set_ylabel(r"90% CI width on $\log S$ (median, IQR)")
    ax_w.set_xlabel("Number of observations $n$")

    fig.savefig(FIG_DIR / "sim_mse_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--render-only", action="store_true",
                        help="Re-render Figure 2 from the saved metrics npz (no refits).")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = FIG_DIR / "sim_mse_coverage_metrics.npz"
    metric_keys = ("wm", "tm", "wc", "tc", "ww", "tw")

    log_f0_common = np.log(true_psd_ls2(U_COMMON, F_COMMON, DT) + 1e-12)
    cal_tang = _tang_calibration()

    # --- Figure 1: single-realization triptych at the largest duration ---
    nt0 = NT_VALUES[-1]
    cal_wdm0 = wdm_white_noise_calibration(nt0 * NF, DT, nt0, WDM_CONFIG)
    data0 = simulate_ls2(nt0 * NF, rng=np.random.default_rng(0))
    rw0, rt0 = _fit_both(data0, nt0, 0, cal_wdm0, cal_tang)
    panels = [
        (log_f0_common, "True $\\log f_0(t,f)$"),
        (_to_common(rw0["time_grid"], rw0["freq_grid"],
                    np.log(rw0["psd_mean"] / cal_wdm0[None, :] + 1e-12)),
         "WDM posterior median"),
        (_to_common(rt0["time_grid"], rt0["freq_grid"],
                    np.log(rt0["psd_mean"] / cal_tang + 1e-12)),
         "Moving-periodogram posterior median"),
    ]
    vmin = min(p.min() for p, _ in panels)
    vmax = max(p.max() for p, _ in panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True, sharey=True)
    for ax, (field, title) in zip(axes, panels):
        mesh = ax.pcolormesh(U_COMMON, F_COMMON, field.T, shading="auto",
                             cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Rescaled time $u$")
    axes[0].set_ylabel("Frequency")
    fig.colorbar(mesh, ax=axes, shrink=0.85, label="$\\log f$")
    fig.savefig(FIG_DIR / "sim_three_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 1 saved (WDM div={rw0['divergences']}, MP div={rt0['divergences']})")

    # --- Figure 2: MSE, coverage, and CI width vs number of observations ---
    # Keep every repeat so we can show the spread (median + interquartile band).
    if args.render_only:
        cached = np.load(metrics_path)
        durations = list(np.asarray(cached["n_total"]))
        raw = {k: list(np.asarray(cached[f"{k}_samples"])) for k in metric_keys}
        _render_metrics(np.asarray(durations), raw)
        print(f"Fig 2 re-rendered from {metrics_path}")
        return

    durations = []
    raw = {k: [] for k in metric_keys}  # per-N lists of arrays
    div_total = 0
    for nt in NT_VALUES:
        n_total = nt * NF
        cal_wdm = wdm_white_noise_calibration(n_total, DT, nt, WDM_CONFIG)
        rep = {k: [] for k in ("wm", "tm", "wc", "tc", "ww", "tw")}
        t0 = time.time()
        for r in range(args.repeats):
            seed = 6000 + r
            data = simulate_ls2(n_total, rng=np.random.default_rng(seed))
            rw, rt = _fit_both(data, nt, seed, cal_wdm, cal_tang)
            div_total += rw["divergences"] + rt["divergences"]
            mw, cw, ww = _metrics(rw, cal_wdm, log_f0_common)
            mt, ct, wt = _metrics(rt, cal_tang, log_f0_common)
            rep["wm"].append(mw); rep["tm"].append(mt)
            rep["wc"].append(cw); rep["tc"].append(ct)
            rep["ww"].append(ww); rep["tw"].append(wt)
        durations.append(n_total)
        for k in raw:
            raw[k].append(np.asarray(rep[k]))
        print(f"n={n_total:6d}  WDM mse={np.median(rep['wm']):.3f} "
              f"cov={np.mean(rep['wc']):.2f} ciw={np.median(rep['ww']):.2f}  "
              f"MP mse={np.median(rep['tm']):.3f} cov={np.mean(rep['tc']):.2f} "
              f"ciw={np.median(rep['tw']):.2f}  ({time.time()-t0:.0f}s)")
    print(f"total divergences: {div_total}  ({args.repeats} repeats/point)")

    durations = np.asarray(durations)
    np.savez(metrics_path, n_total=durations, repeats=args.repeats,
             **{f"{k}_samples": np.stack(raw[k]) for k in raw})
    _render_metrics(durations, raw)
    print(f"Fig 2 saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
