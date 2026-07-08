"""Two-likelihood LS2 simulation study figures (manuscript Section 4).

Both observation models -- the WDM coefficient likelihood and the Tang zigzag
moving-periodogram dynamic Whittle -- are fitted with the *same* whitened
P-spline model, so the comparison isolates the time-frequency representation.

Produces:
  * ``sim_three_panel.png``  (Fig 1) -- true log-PSD, WDM posterior median, and
    moving-periodogram posterior median for a single LS2 realization, shared
    colour scale.
  * ``sim_mse_coverage.png`` (Fig 2) -- MSE_{log f}, 90% credible-interval
    coverage, CI width, and per-fit wall time versus the number of
    observations, one curve per likelihood, each point over ``--repeats``
    realizations.

Metrics are saved as one shard per duration (``sim_metrics_nt{N}.npz``), so
cluster array jobs can each run a single ``--nt`` value and the figure is
re-rendered from all shards with ``--render-only``. Every render also writes a
long-format ``sim_metrics.csv`` (one row per realization) next to the shards --
edit values there and pass ``--from-csv`` to re-render Fig 2 from the edited
file without touching the npz shards.

    python notes/scripts/make_sim_study_figures.py --repeats 20
    python notes/scripts/make_sim_study_figures.py --nt 384 --skip-fig1  # one shard
    python notes/scripts/make_sim_study_figures.py --render-only
    python notes/scripts/make_sim_study_figures.py --from-csv notes/figures/sim_metrics.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from datasets import simulate_ls2, true_psd_ls2, wdm_white_noise_calibration
from tv_pspline_psd import (
    PSplineConfig,
    interval_coverage,
    run_tang_dynamic_whittle_mcmc,
    run_wdm_psd_mcmc,
    set_paper_style,
    summarize_mcmc_diagnostics,
    tang_moving_periodogram,
)

set_paper_style()

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT = 0.1
NF = 24
NT_VALUES = (24, 48, 96, 192, 384)
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


# Two chains of 500/500: single 250-draw chains leave rhat(phi) ~ 1.1 at the
# largest durations, and sampling is <1 s per fit so the longer chains are free.
def _fit_both(data, nt, seed, cal_wdm, cal_tang):
    rw = run_wdm_psd_mcmc(data, dt=DT, nt=nt, config=WDM_CONFIG,
                          n_warmup=500, n_samples=500, num_chains=2,
                          random_seed=seed)
    rt = run_tang_dynamic_whittle_mcmc(data, dt=DT, m=TANG_M, thin=TANG_THIN,
                                       config=TANG_CONFIG, n_warmup=500,
                                       n_samples=500, num_chains=2,
                                       random_seed=seed)
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


METRIC_KEYS = ("wm", "tm", "wc", "tc", "ww", "tw", "wt", "tt", "wr", "tr", "we", "te")
# Metric-key naming: prefix selects the likelihood, suffix the quantity.
LIKELIHOOD_PREFIX = {"w": "wdm", "t": "mp"}
METRIC_SUFFIX = {"m": "mse", "c": "coverage", "w": "ci_width",
                 "t": "wall_time_s", "r": "rhat", "e": "neff"}
CSV_COLUMNS = ("n_total", "likelihood", "repeat", *METRIC_SUFFIX.values())


def _diag_extrema(res) -> tuple[float, float]:
    """Max r_hat / min n_eff over the smoothing-precision sites."""
    d = summarize_mcmc_diagnostics(res)
    return (max(d["phi_time"]["r_hat"], d["phi_freq"]["r_hat"]),
            min(d["phi_time"]["n_eff"], d["phi_freq"]["n_eff"]))


def _shard_path(n_total: int) -> Path:
    return FIG_DIR / f"sim_metrics_nt{n_total:05d}.npz"


def _load_shards() -> tuple[np.ndarray, dict[str, list]]:
    shards = sorted(FIG_DIR.glob("sim_metrics_nt*.npz"))
    if not shards:
        raise FileNotFoundError(f"no sim_metrics_nt*.npz shards in {FIG_DIR}")
    durations, raw = [], {k: [] for k in METRIC_KEYS}
    for path in shards:
        with np.load(path) as f:
            durations.append(int(f["n_total"]))
            for k in METRIC_KEYS:
                raw[k].append(np.asarray(f[f"{k}_samples"]))
    return np.asarray(durations), raw


def _raw_to_dataframe(durations: np.ndarray, raw: dict[str, list]) -> pd.DataFrame:
    """Long-format table, one row per (duration, likelihood, repeat)."""
    rows = []
    for i, n_total in enumerate(durations):
        for prefix, likelihood in LIKELIHOOD_PREFIX.items():
            columns = {name: np.asarray(raw[f"{prefix}{suffix}"][i])
                      for suffix, name in METRIC_SUFFIX.items()}
            n_repeats = len(next(iter(columns.values())))
            for r in range(n_repeats):
                rows.append({"n_total": int(n_total), "likelihood": likelihood,
                            "repeat": r, **{k: float(v[r]) for k, v in columns.items()}})
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def _dataframe_to_raw(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, list]]:
    """Inverse of :func:`_raw_to_dataframe`, for re-rendering from an edited CSV."""
    durations = np.sort(df["n_total"].unique())
    inv_prefix = {v: k for k, v in LIKELIHOOD_PREFIX.items()}
    raw = {k: [] for k in METRIC_KEYS}
    for n_total in durations:
        sub = df[df["n_total"] == n_total]
        for likelihood, prefix in inv_prefix.items():
            block = sub[sub["likelihood"] == likelihood].sort_values("repeat")
            for suffix, name in METRIC_SUFFIX.items():
                raw[f"{prefix}{suffix}"].append(block[name].to_numpy(dtype=float))
    return durations, raw


def _render_metrics(durations: np.ndarray, raw: dict[str, list]) -> None:
    """Render the MSE / coverage / CI-width / runtime vs. n figure."""
    med = lambda key: np.array([np.median(a) for a in raw[key]])
    q = lambda key, p: np.array([np.percentile(a, p) for a in raw[key]])

    def _band(ax, key, color, marker, label):
        ax.plot(durations, med(key), marker, color=color, label=label)
        ax.fill_between(durations, q(key, 25), q(key, 75), color=color, alpha=0.18)

    from matplotlib.ticker import FixedLocator, NullFormatter

    fig, (ax_m, ax_c, ax_w, ax_t) = plt.subplots(4, 1, figsize=(3.6, 6.2),
                                                 sharex=True,
                                                 constrained_layout=True)
    _band(ax_m, "wm", "tab:blue", "o-", "WDM")
    _band(ax_m, "tm", "tab:orange", "s--", "Moving periodogram")
    ax_m.set_xscale("log"); ax_m.set_yscale("log")
    ax_m.set_ylabel(r"$\mathrm{MSE}_{\log f}$")
    ax_m.legend(fontsize=8)

    ax_c.semilogx(durations, np.array([np.mean(a) for a in raw["wc"]]), "o-", color="tab:blue")
    ax_c.semilogx(durations, np.array([np.mean(a) for a in raw["tc"]]), "s--", color="tab:orange")
    ax_c.axhline(0.9, ls=":", color="black", label="nominal 90%")
    ax_c.set_ylim(0.0, 1.0); ax_c.set_ylabel(r"$90\%$ coverage"); ax_c.legend(fontsize=8)

    _band(ax_w, "ww", "tab:blue", "o-", "WDM")
    _band(ax_w, "tw", "tab:orange", "s--", "Moving periodogram")
    ax_w.set_xscale("log"); ax_w.set_yscale("log")
    ax_w.set_ylabel(r"$90\%$ CI width")

    _band(ax_t, "wt", "tab:blue", "o-", "WDM")
    _band(ax_t, "tt", "tab:orange", "s--", "Moving periodogram")
    ax_t.set_xscale("log"); ax_t.set_yscale("log")
    ax_t.set_ylabel("Wall time [s]")
    ax_t.set_xlabel("Number of observations $n$")

    # Explicit ticks at the sampled lengths; the sub-decade span otherwise
    # auto-labels crowded minor ticks.
    ax_t.xaxis.set_major_locator(FixedLocator(list(durations)))
    ax_t.xaxis.set_minor_formatter(NullFormatter())
    ax_t.set_xticklabels([f"{int(d)}" for d in durations])

    fig.savefig(FIG_DIR / "sim_mse_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = FIG_DIR / "sim_metrics.csv"
    _raw_to_dataframe(durations, raw).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--nt", type=int, nargs="*", default=list(NT_VALUES),
                        help="Time-bin counts to fit (each saved as its own shard); "
                             "pass with no values to make Fig 1 and re-render only.")
    parser.add_argument("--skip-fig1", action="store_true",
                        help="Skip the single-realization triptych (for shard jobs).")
    parser.add_argument("--render-only", action="store_true",
                        help="Re-render Figure 2 from the saved shards (no refits).")
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="Re-render Figure 2 from a (possibly hand-edited) "
                             "sim_metrics.csv instead of the npz shards.")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv is not None:
        durations, raw = _dataframe_to_raw(pd.read_csv(args.from_csv))
        _render_metrics(durations, raw)
        print(f"Fig 2 re-rendered from {args.from_csv}")
        return

    if args.render_only:
        durations, raw = _load_shards()
        _render_metrics(durations, raw)
        print(f"Fig 2 re-rendered from {len(durations)} shards in {FIG_DIR}")
        return

    log_f0_common = np.log(true_psd_ls2(U_COMMON, F_COMMON, DT) + 1e-12)
    cal_tang = _tang_calibration()

    # --- Figure 1: single-realization triptych at the largest duration ---
    if not args.skip_fig1:
        nt0 = NT_VALUES[-1]
        cal_wdm0 = wdm_white_noise_calibration(nt0 * NF, DT, nt0, WDM_CONFIG)
        data0 = simulate_ls2(nt0 * NF, rng=np.random.default_rng(0))
        rw0, rt0 = _fit_both(data0, nt0, 0, cal_wdm0, cal_tang)
        panels = [
            (log_f0_common, r"Truth: $\log f_0(t,f)$"),
            (_to_common(rw0["time_grid"], rw0["freq_grid"],
                        np.log(rw0["psd_mean"] / cal_wdm0[None, :] + 1e-12)),
             "WDM"),
            (_to_common(rt0["time_grid"], rt0["freq_grid"],
                        np.log(rt0["psd_mean"] / cal_tang + 1e-12)),
             "Moving periodogram"),
        ]
        vmin = min(p.min() for p, _ in panels)
        vmax = max(p.max() for p, _ in panels)
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), constrained_layout=True, sharey=True)
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

    # --- Figure 2: MSE, coverage, CI width, and runtime vs observations ---
    # Keep every repeat so we can show the spread (median + interquartile band).
    div_total = 0
    for nt in args.nt:
        n_total = nt * NF
        cal_wdm = wdm_white_noise_calibration(n_total, DT, nt, WDM_CONFIG)
        rep = {k: [] for k in METRIC_KEYS}
        t0 = time.time()
        for r in range(args.repeats):
            seed = 6000 + r
            data = simulate_ls2(n_total, rng=np.random.default_rng(seed))
            rw, rt = _fit_both(data, nt, seed, cal_wdm, cal_tang)
            div_total += rw["divergences"] + rt["divergences"]
            mw, cw, ww = _metrics(rw, cal_wdm, log_f0_common)
            mt, ct, wt = _metrics(rt, cal_tang, log_f0_common)
            rhat_w, neff_w = _diag_extrema(rw)
            rhat_t, neff_t = _diag_extrema(rt)
            rep["wm"].append(mw); rep["tm"].append(mt)
            rep["wc"].append(cw); rep["tc"].append(ct)
            rep["ww"].append(ww); rep["tw"].append(wt)
            rep["wt"].append(rw["nuts_runtime_s"]); rep["tt"].append(rt["nuts_runtime_s"])
            rep["wr"].append(rhat_w); rep["tr"].append(rhat_t)
            rep["we"].append(neff_w); rep["te"].append(neff_t)
        np.savez(_shard_path(n_total), n_total=n_total, repeats=args.repeats,
                 **{f"{k}_samples": np.asarray(rep[k]) for k in METRIC_KEYS})
        print(f"n={n_total:6d}  WDM mse={np.median(rep['wm']):.3f} "
              f"cov={np.mean(rep['wc']):.2f} ciw={np.median(rep['ww']):.2f} "
              f"t={np.median(rep['wt']):.1f}s rhat<={np.max(rep['wr']):.3f} "
              f"neff>={np.min(rep['we']):.0f}  "
              f"MP mse={np.median(rep['tm']):.3f} cov={np.mean(rep['tc']):.2f} "
              f"ciw={np.median(rep['tw']):.2f} t={np.median(rep['tt']):.1f}s "
              f"rhat<={np.max(rep['tr']):.3f} neff>={np.min(rep['te']):.0f}  "
              f"({time.time()-t0:.0f}s)")
    print(f"total divergences: {div_total}  ({args.repeats} repeats/point)")

    durations, raw = _load_shards()
    _render_metrics(durations, raw)
    print(f"Fig 2 saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
