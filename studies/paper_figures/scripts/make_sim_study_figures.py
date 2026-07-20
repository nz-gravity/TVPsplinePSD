"""Two-likelihood LS2 simulation study figures (manuscript Section 4).

Both observation models -- the WDM coefficient likelihood and the Tang zigzag
moving-periodogram dynamic Whittle -- are fitted with the *same* whitened
P-spline model, so the comparison isolates the time-frequency representation.

Produces:
  * ``sim_three_panel.png``  (Fig 2) -- true log-PSD and posterior geometric
    means from WDM and the moving periodogram for one realization, with a shared
    colour scale.
  * ``sim_mse_coverage.png`` (Fig 3) -- MSE_{log f}, 90% credible-interval
    coverage, CI width, and per-fit wall time versus the number of
    observations, one curve per likelihood, each point over ``--repeats``
    realizations.

Metrics are saved as one shard per duration and common knot count
(``sim_metrics_kf{K}_nt{N}.npz``), so cluster array jobs can run one pair at a
time and the figure can be re-rendered with ``--render-only``. Every render also
writes a long-format CSV (one row per realization) next to the shards; pass it
back with ``--from-csv`` to re-render Figure 3 without refitting.

    python studies/paper_figures/scripts/make_sim_study_figures.py --repeats 20
    python studies/paper_figures/scripts/make_sim_study_figures.py --nt 384 --skip-fig1  # one shard
    python studies/paper_figures/scripts/make_sim_study_figures.py --render-only
    python studies/paper_figures/scripts/make_sim_study_figures.py --from-csv studies/paper_figures/figures/sim_metrics.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tv_pspline_psd import (
    PSplineConfig,
    run_tang_dynamic_whittle_mcmc,
    run_wdm_psd_mcmc,
    set_paper_style,
    summarize_mcmc_diagnostics,
    tang_moving_periodogram,
)
from tv_pspline_psd.datasets import (
    monte_carlo_reference,
    simulate_ls2,
    true_psd_ls2,
    wdm_white_noise_calibration,
)
from tv_pspline_psd.inference import reconstruct_eig_coeff_samples
from tv_pspline_psd.splines import evaluate_bspline_basis

set_paper_style()

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT = 0.1
# Number of WDM frequency channels.  The production power-of-two sweep sets
# this to 32 so every total series length can itself be a power of two.
NF = 24
NT_VALUES = (24, 48, 96, 192, 384, 768, 1536)
TANG_M, TANG_THIN = 16, 2
# Fixed interior evaluation domain shared by every duration and both front
# ends.  [0.10, 0.85] remains inside the trimmed WDM support of the smallest
# production grid (nt=16, N=512, nf=32).
U_COMMON = np.linspace(0.10, 0.85, 60)
F_COMMON = np.linspace(0.6, 4.4, 60)
DEFAULT_FREQ_KNOTS = 8
KNOT_SENSITIVITY = (6, 8, 10)
REFERENCE_CACHE_VERSION = 1


def _time_knots(nt: int) -> int:
    """Interior time knots growing as nt^{2/5} (anchored at 8 for nt=24).

    Only the time axis gains data as n grows (nf is fixed), so a fixed basis
    leaves a residual smoothing bias that the contracting credible intervals
    stop covering; growing the knot count with n restores nominal coverage.
    """
    return round(8 * (nt / 24) ** 0.4)


def _configs(nt: int, freq_knots: int) -> tuple[PSplineConfig, PSplineConfig]:
    kt = _time_knots(nt)
    common = dict(
        n_interior_knots_time=kt,
        n_interior_knots_freq=freq_knots,
        freq_knot_strategy="linear",
    )
    return PSplineConfig(**common), PSplineConfig(**common)


def _tang_calibration() -> float:
    # The moving Fourier coefficient is divided by sqrt(2*pi*(2m+1)); for
    # unit-variance white noise E[MI] is therefore exactly 1/(2*pi).
    return 1.0 / (2.0 * np.pi)


# Both front ends use exactly the same interior-knot counts and physical
# locations. Their clamped boundary knots still follow the support of their
# respective likelihood grids, while every reported metric is evaluated well
# inside the common support below.
def _common_interior_knots(nt: int, freq_knots: int) -> tuple[np.ndarray, np.ndarray]:
    # WDM support after the default one-bin trims is [1/nt, (nt-2)/nt].
    # Tang's windowed support is slightly wider for every duration here.
    time = np.linspace(1.0 / nt, (nt - 2.0) / nt, _time_knots(nt) + 2)[1:-1]
    # Common frequency support after WDM trimming and over Tang's m rungs.
    wdm_low = 1.0 / (2.0 * NF * DT)
    wdm_high = (NF - 1.0) / (2.0 * NF * DT)
    tang_low = 1.0 / ((2.0 * TANG_M + 1.0) * DT)
    tang_high = TANG_M / ((2.0 * TANG_M + 1.0) * DT)
    freq = np.linspace(max(wdm_low, tang_low), min(wdm_high, tang_high),
                       freq_knots + 2)[1:-1]
    return time, freq


# Two chains of 500/500: single 250-draw chains leave rhat(phi) ~ 1.1 at the
# largest durations. Longer chains do not materially change the surface MSE.
def _fit_both(data, nt, seed, freq_knots):
    cfg_wdm, cfg_tang = _configs(nt, freq_knots)
    knots_time, knots_freq = _common_interior_knots(nt, freq_knots)
    rw = run_wdm_psd_mcmc(data, dt=DT, nt=nt, config=cfg_wdm,
                          interior_knots_time=knots_time,
                          interior_knots_freq=knots_freq,
                          n_warmup=500, n_samples=500, num_chains=2,
                          random_seed=seed)
    rt = run_tang_dynamic_whittle_mcmc(data, dt=DT, m=TANG_M, thin=TANG_THIN,
                                       config=cfg_tang,
                                       interior_knots_time=knots_time,
                                       interior_knots_freq=knots_freq,
                                       n_warmup=500,
                                       n_samples=500, num_chains=2,
                                       random_seed=seed)
    return rw, rt


def _eig_samples(res, representation):
    if representation == "wdm":
        return reconstruct_eig_coeff_samples(res["samples"], res["whitened"], res["config"])
    return np.asarray(res["eig_coeff_samples"])


def _basis_pair(res, time, freq, representation):
    degree_t = res["config"].degree_time
    degree_f = res["config"].degree_freq
    if representation == "wdm":
        freq_coordinate = np.asarray(freq) / np.asarray(res["freq_grid"])[-1]
    else:
        freq_coordinate = 2.0 * DT * np.asarray(freq)
    bt = evaluate_bspline_basis(time, res["knots_time"], degree=degree_t)
    bf = evaluate_bspline_basis(freq_coordinate, res["knots_freq"], degree=degree_f)
    return bt @ res["whitened"]["U_time"], bf @ res["whitened"]["U_freq"]


def _log_surface_samples(res, time, freq, representation):
    bt, bf = _basis_pair(res, time, freq, representation)
    return np.einsum("ta,nab,fb->ntf", bt, _eig_samples(res, representation), bf,
                     optimize=True)


def _log_surface_mean_at_points(res, time, freq, representation):
    bt, bf = _basis_pair(res, time, freq, representation)
    return np.einsum("pa,ab,pb->p", bt, _eig_samples(res, representation).mean(axis=0),
                     bf, optimize=True)


def _metrics(res, cal, log_f0_common, representation, native_reference):
    cal = np.asarray(cal)
    log_draws = _log_surface_samples(res, U_COMMON, F_COMMON, representation)
    if cal.ndim == 0:
        log_cal = np.log(cal)
    else:
        log_cal = np.interp(F_COMMON, res["freq_grid"], np.log(cal))[None, :]
    log_draws = log_draws - log_cal
    fitted = log_draws.mean(axis=0)
    mse = float(np.mean((fitted - log_f0_common) ** 2))
    lower, upper = np.percentile(log_draws, [5.0, 95.0], axis=0)
    cov = float(np.mean((log_f0_common >= lower) & (log_f0_common <= upper)))
    ci_width = float(np.mean(upper - lower))

    # Diagnostic MSE against the exact finite-resolution estimand of the front
    # end. This is deliberately separate from the common latent-PSD MSE above.
    if representation == "wdm":
        native_fit = np.asarray(res["log_psd_mean"])
    else:
        ordinates = res["ordinates"]
        native_fit = _log_surface_mean_at_points(
            res,
            np.asarray(ordinates["u"]),
            np.asarray(ordinates["omega"]) / (2.0 * np.pi * DT),
            representation,
        )
    native_mse = float(np.mean((native_fit - np.log(native_reference)) ** 2))
    return mse, cov, ci_width, native_mse


METRIC_KEYS = ("wm", "tm", "wc", "tc", "ww", "tw", "wn", "tn",
               "wt", "tt", "wr", "tr", "we", "te")
# Metric-key naming: prefix selects the likelihood, suffix the quantity.
LIKELIHOOD_PREFIX = {"w": "wdm", "t": "mp"}
METRIC_SUFFIX = {"m": "mse", "c": "coverage", "w": "ci_width",
                 "n": "native_mse",
                 "t": "wall_time_s", "r": "rhat", "e": "neff"}
CSV_COLUMNS = ("n_total", "freq_knots", "likelihood", "repeat", *METRIC_SUFFIX.values())


def _diag_extrema(res) -> tuple[float, float]:
    """Max r_hat / min n_eff over the smoothing-precision sites."""
    d = summarize_mcmc_diagnostics(res)
    return (max(d["phi_time"]["r_hat"], d["phi_freq"]["r_hat"]),
            min(d["phi_time"]["n_eff"], d["phi_freq"]["n_eff"]))


def _shard_path(n_total: int, freq_knots: int) -> Path:
    prefix = "sim_metrics" if NF == 24 else f"sim_metrics_nf{NF:02d}"
    return FIG_DIR / f"{prefix}_kf{freq_knots:02d}_nt{n_total:05d}.npz"


def _chunk_path(
    n_total: int, freq_knots: int, repeat_start: int, repeats: int
) -> Path:
    repeat_stop = repeat_start + repeats
    prefix = "sim_metrics" if NF == 24 else f"sim_metrics_nf{NF:02d}"
    return (
        FIG_DIR / "chunks" /
        f"{prefix}_kf{freq_knots:02d}_nt{n_total:05d}_"
        f"r{repeat_start:03d}-{repeat_stop - 1:03d}.npz"
    )


def _atomic_savez(path: Path, **arrays) -> None:
    """Atomically replace an NPZ checkpoint on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_cache_path(
    cache_dir: Path, nt: int, freq_knots: int, n_draws: int
) -> Path:
    return cache_dir / (
        f"ls2_refs_v{REFERENCE_CACHE_VERSION}_nf{NF:02d}_nt{nt:04d}_"
        f"kf{freq_knots:02d}_nd{n_draws:04d}.npz"
    )


def _reference_metadata(nt: int, freq_knots: int, n_draws: int) -> dict[str, object]:
    return {
        "cache_version": REFERENCE_CACHE_VERSION,
        "dt": DT,
        "nf": NF,
        "nt": nt,
        "n_total": nt * NF,
        "freq_knots": freq_knots,
        "time_knots": _time_knots(nt),
        "tang_m": TANG_M,
        "tang_thin": TANG_THIN,
        "reference_draws": n_draws,
    }


def _read_reference_cache(
    path: Path, nt: int, freq_knots: int, n_draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = _reference_metadata(nt, freq_knots, n_draws)
    with np.load(path) as saved:
        for key, value in expected.items():
            actual = saved[key].item()
            if actual != value:
                raise ValueError(
                    f"Reference cache {path} has {key}={actual!r}; expected {value!r}."
                )
        return (
            np.asarray(saved["cal_wdm"]),
            np.asarray(saved["ref_wdm"]),
            np.asarray(saved["ref_tang"]),
        )


def _load_or_create_references(
    nt: int,
    freq_knots: int,
    n_draws: int,
    cache_dir: Path,
    *,
    require_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _reference_cache_path(cache_dir, nt, freq_knots, n_draws)
    if path.exists():
        values = _read_reference_cache(path, nt, freq_knots, n_draws)
        print(f"Loaded reference cache {path}", flush=True)
        return values
    if require_cache:
        raise FileNotFoundError(
            f"Required reference cache is missing: {path}. Run the reference-preparation "
            "Slurm array first."
        )

    cfg_wdm, _ = _configs(nt, freq_knots)
    cal_wdm = wdm_white_noise_calibration(
        nt * NF, DT, nt, cfg_wdm, n_draws=n_draws
    )
    ref_wdm, ref_tang = _finite_resolution_references(nt, cfg_wdm, n_draws)
    _atomic_savez(
        path,
        **_reference_metadata(nt, freq_knots, n_draws),
        cal_wdm=cal_wdm,
        ref_wdm=ref_wdm,
        ref_tang=ref_tang,
    )
    print(f"Wrote reference cache {path}", flush=True)
    return np.asarray(cal_wdm), np.asarray(ref_wdm), np.asarray(ref_tang)


def _load_shards(freq_knots: int) -> tuple[np.ndarray, dict[str, list]]:
    prefix = "sim_metrics" if NF == 24 else f"sim_metrics_nf{NF:02d}"
    shards = sorted(FIG_DIR.glob(f"{prefix}_kf{freq_knots:02d}_nt*.npz"))
    if not shards:
        raise FileNotFoundError(
            f"no matched-knot nf={NF} shards for {freq_knots} frequency knots in {FIG_DIR}"
        )
    durations, raw = [], {k: [] for k in METRIC_KEYS}
    for path in shards:
        with np.load(path) as f:
            durations.append(int(f["n_total"]))
            for k in METRIC_KEYS:
                raw[k].append(np.asarray(f[f"{k}_samples"]))
    return np.asarray(durations), raw


def _raw_to_dataframe(
    durations: np.ndarray, raw: dict[str, list], freq_knots: int
) -> pd.DataFrame:
    """Long-format table, one row per (duration, likelihood, repeat)."""
    rows = []
    for i, n_total in enumerate(durations):
        for prefix, likelihood in LIKELIHOOD_PREFIX.items():
            columns = {name: np.asarray(raw[f"{prefix}{suffix}"][i])
                      for suffix, name in METRIC_SUFFIX.items()}
            n_repeats = len(next(iter(columns.values())))
            for r in range(n_repeats):
                rows.append({"n_total": int(n_total), "freq_knots": int(freq_knots),
                            "likelihood": likelihood,
                            "repeat": r, **{k: float(v[r]) for k, v in columns.items()}})
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def _dataframe_to_raw(
    df: pd.DataFrame, freq_knots: int
) -> tuple[np.ndarray, dict[str, list]]:
    """Inverse of :func:`_raw_to_dataframe`, for re-rendering from an edited CSV."""
    if "freq_knots" in df:
        df = df[df["freq_knots"] == freq_knots]
    if df.empty:
        raise ValueError(f"CSV has no rows for freq_knots={freq_knots}.")
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


def _render_metrics(
    durations: np.ndarray, raw: dict[str, list], freq_knots: int
) -> None:
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
    ax_t.set_xticklabels([f"{int(d)}" for d in durations], rotation=35, ha="right")
    ax_t.tick_params(axis="x", labelsize=7)

    fig.savefig(FIG_DIR / "sim_mse_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Diagnostic decomposition: the common analytic target is the primary
    # cross-method comparison; native expected-power targets show how much of
    # each error comes from the representation rather than the surface fit.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, common_key, native_key, title in (
        (axes[0], "wm", "wn", "WDM"),
        (axes[1], "tm", "tn", "Moving periodogram"),
    ):
        ax.loglog(durations, med(common_key), "o-", label="analytic PSD")
        ax.loglog(durations, med(native_key), "s--", label="native expected power")
        ax.set_title(title)
        ax.set_xlabel("Number of observations $n$")
    axes[0].set_ylabel(r"$\mathrm{MSE}_{\log f}$")
    axes[1].legend(fontsize=8)
    fig.savefig(
        FIG_DIR / f"sim_mse_targets_kf{freq_knots:02d}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    csv_path = FIG_DIR / f"sim_metrics_kf{freq_knots:02d}.csv"
    frame = _raw_to_dataframe(durations, raw, freq_knots)
    frame.to_csv(csv_path, index=False)
    if freq_knots == DEFAULT_FREQ_KNOTS:
        frame.to_csv(FIG_DIR / "sim_metrics.csv", index=False)
    print(f"Wrote {csv_path}")


def _render_knot_sensitivity() -> None:
    """Render a diagnostic comparison across every complete knot-count study."""
    loaded = {}
    for knots in KNOT_SENSITIVITY:
        try:
            loaded[knots] = _load_shards(knots)
        except FileNotFoundError:
            continue
    if len(loaded) < 2:
        return
    duration_sets = {tuple(durations) for durations, _ in loaded.values()}
    if len(duration_sets) != 1:
        # A staged pilot may deliberately run knot sensitivity only at the
        # shortest and longest records. Do not render an apples-to-oranges
        # multi-knot curve until every available count shares the same n grid.
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True,
                             sharex=True, sharey=True)
    colors = dict(zip(KNOT_SENSITIVITY, ("tab:green", "tab:blue", "tab:purple")))
    for knots, (durations, raw) in loaded.items():
        for ax, key, title in ((axes[0], "wm", "WDM"), (axes[1], "tm", "Moving periodogram")):
            med = np.array([np.median(a) for a in raw[key]])
            ax.loglog(durations, med, "o-", color=colors[knots], label=f"{knots} knots")
            ax.set_title(title)
            ax.set_xlabel("Number of observations $n$")
    axes[0].set_ylabel(r"$\mathrm{MSE}_{\log f}$")
    axes[1].legend(fontsize=8)
    fig.savefig(FIG_DIR / "sim_knot_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _finite_resolution_references(nt, config, n_draws):
    n_total = nt * NF
    wdm_reference = monte_carlo_reference(
        lambda rng: simulate_ls2(n_total, rng=rng),
        n_draws=n_draws,
        n_total=n_total,
        dt=DT,
        nt=nt,
        config=config,
        seed=8100 + nt,
    )
    rng = np.random.default_rng(9100 + nt)
    moving_reference = None
    for _ in range(n_draws):
        mi = tang_moving_periodogram(
            simulate_ls2(n_total, rng=rng), m=TANG_M, thin=TANG_THIN
        )["mi"]
        if moving_reference is None:
            moving_reference = np.zeros_like(mi)
        moving_reference += mi
    return wdm_reference, moving_reference / n_draws


def _checkpoint_arrays(
    rep: dict[str, list],
    *,
    n_total: int,
    freq_knots: int,
    repeat_start: int,
    repeats_target: int,
) -> dict[str, object]:
    completed = len(rep[METRIC_KEYS[0]])
    return {
        "n_total": n_total,
        "nf": NF,
        "freq_knots": freq_knots,
        "repeat_start": repeat_start,
        "repeats_target": repeats_target,
        "repeat_ids": np.arange(repeat_start, repeat_start + completed),
        **{f"{key}_samples": np.asarray(rep[key]) for key in METRIC_KEYS},
    }


def _load_checkpoint(
    path: Path,
    *,
    n_total: int,
    freq_knots: int,
    repeat_start: int,
    repeats_target: int,
) -> dict[str, list]:
    rep = {key: [] for key in METRIC_KEYS}
    if not path.exists():
        return rep
    expected = {
        "n_total": n_total,
        "nf": NF,
        "freq_knots": freq_knots,
        "repeat_start": repeat_start,
        "repeats_target": repeats_target,
    }
    with np.load(path) as saved:
        for key, value in expected.items():
            actual = int(saved[key])
            if actual != value:
                raise ValueError(
                    f"Checkpoint {path} has {key}={actual}; expected {value}."
                )
        repeat_ids = np.asarray(saved["repeat_ids"], dtype=int)
        wanted = np.arange(repeat_start, repeat_start + repeat_ids.size)
        if not np.array_equal(repeat_ids, wanted):
            raise ValueError(f"Checkpoint {path} has non-contiguous repeat IDs.")
        if repeat_ids.size > repeats_target:
            raise ValueError(f"Checkpoint {path} contains too many repeats.")
        for key in METRIC_KEYS:
            values = np.asarray(saved[f"{key}_samples"])
            if values.size != repeat_ids.size:
                raise ValueError(f"Checkpoint {path} has inconsistent metric lengths.")
            rep[key] = values.tolist()
    print(
        f"Resuming {path} after {len(rep[METRIC_KEYS[0]])}/{repeats_target} repeats",
        flush=True,
    )
    return rep


def _merge_chunks(
    nt_values: list[int], freq_knots: int, total_repeats: int, chunk_size: int
) -> None:
    if total_repeats % chunk_size:
        raise ValueError("total repeats must be divisible by chunk size")
    for nt in nt_values:
        n_total = nt * NF
        merged = {key: [] for key in METRIC_KEYS}
        merged_ids = []
        for start in range(0, total_repeats, chunk_size):
            path = _chunk_path(n_total, freq_knots, start, chunk_size)
            rep = _load_checkpoint(
                path,
                n_total=n_total,
                freq_knots=freq_knots,
                repeat_start=start,
                repeats_target=chunk_size,
            )
            if len(rep[METRIC_KEYS[0]]) != chunk_size:
                raise ValueError(
                    f"Chunk {path} is incomplete: "
                    f"{len(rep[METRIC_KEYS[0]])}/{chunk_size} repeats."
                )
            merged_ids.extend(range(start, start + chunk_size))
            for key in METRIC_KEYS:
                merged[key].extend(rep[key])
        if merged_ids != list(range(total_repeats)):
            raise ValueError(f"Merged repeat IDs are incomplete for n={n_total}.")
        output = _shard_path(n_total, freq_knots)
        _atomic_savez(
            output,
            n_total=n_total,
            nf=NF,
            freq_knots=freq_knots,
            repeats=total_repeats,
            repeat_ids=np.asarray(merged_ids),
            **{f"{key}_samples": np.asarray(merged[key]) for key in METRIC_KEYS},
        )
        print(f"Merged {total_repeats} repeats into {output}", flush=True)


def main() -> None:
    global NF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--freq-knots", type=int, default=DEFAULT_FREQ_KNOTS,
                        help="Common interior frequency-knot count for both front ends.")
    parser.add_argument("--nf", type=int, default=NF,
                        help="Number of WDM frequency channels (must be even).")
    parser.add_argument("--reference-draws", type=int, default=1000,
                        help="Monte Carlo draws for each finite-resolution target.")
    parser.add_argument("--reference-cache-dir", type=Path,
                        default=FIG_DIR / "reference_cache",
                        help="Directory containing deterministic per-size references.")
    parser.add_argument("--prepare-references", action="store_true",
                        help="Create/load reference caches for --nt and exit.")
    parser.add_argument("--require-reference-cache", action="store_true",
                        help="Fail instead of computing a missing reference cache.")
    parser.add_argument("--nt", type=int, nargs="*", default=list(NT_VALUES),
                        help="Time-bin counts to fit (each saved as its own shard); "
                             "pass with no values to make the single-realisation panel.")
    parser.add_argument("--repeat-start", type=int, default=None,
                        help="Global first repeat ID; enables resumable chunk output.")
    parser.add_argument("--merge-chunks", action="store_true",
                        help="Merge complete chunk checkpoints for --nt and exit.")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Repeats per chunk when using --merge-chunks.")
    parser.add_argument("--skip-fig1", action="store_true",
                        help="Skip the single-realization triptych (for shard jobs).")
    parser.add_argument("--skip-render", action="store_true",
                        help="Write requested shards without rendering shared figures.")
    parser.add_argument("--render-only", action="store_true",
                        help="Re-render Figure 3 from the saved shards (no refits).")
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="Re-render Figure 3 from a (possibly hand-edited) "
                             "sim_metrics.csv instead of the npz shards.")
    args = parser.parse_args()
    if args.freq_knots < 1:
        parser.error("--freq-knots must be positive")
    if args.nf < 2 or args.nf % 2:
        parser.error("--nf must be an even integer of at least 2")
    if args.reference_draws < 1:
        parser.error("--reference-draws must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.repeat_start is not None and args.repeat_start < 0:
        parser.error("--repeat-start must be non-negative")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    NF = args.nf
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.prepare_references:
        for nt in args.nt:
            _load_or_create_references(
                nt,
                args.freq_knots,
                args.reference_draws,
                args.reference_cache_dir,
                require_cache=False,
            )
        return

    if args.merge_chunks:
        _merge_chunks(args.nt, args.freq_knots, args.repeats, args.chunk_size)
        return

    if args.from_csv is not None:
        durations, raw = _dataframe_to_raw(pd.read_csv(args.from_csv), args.freq_knots)
        _render_metrics(durations, raw, args.freq_knots)
        _render_knot_sensitivity()
        print(f"Figure 3 re-rendered from {args.from_csv}")
        return

    if args.render_only:
        durations, raw = _load_shards(args.freq_knots)
        _render_metrics(durations, raw, args.freq_knots)
        _render_knot_sensitivity()
        print(f"Figure 3 re-rendered from {len(durations)} shards in {FIG_DIR}")
        return

    log_f0_common = np.log(true_psd_ls2(U_COMMON, F_COMMON, DT) + 1e-12)
    cal_tang = _tang_calibration()

    # --- Single-realization triptych at the largest duration. ---
    if not args.skip_fig1:
        nt0 = NT_VALUES[-1]
        cfg_wdm0, _ = _configs(nt0, args.freq_knots)
        cal_wdm0 = wdm_white_noise_calibration(
            nt0 * NF, DT, nt0, cfg_wdm0, n_draws=args.reference_draws
        )
        data0 = simulate_ls2(nt0 * NF, rng=np.random.default_rng(0))
        rw0, rt0 = _fit_both(data0, nt0, 0, args.freq_knots)
        panels = [
            (log_f0_common, r"Truth: $\log f_0(t,f)$"),
            (_log_surface_samples(rw0, U_COMMON, F_COMMON, "wdm").mean(axis=0)
             - np.interp(F_COMMON, rw0["freq_grid"], np.log(cal_wdm0))[None, :],
             "WDM"),
            (_log_surface_samples(rt0, U_COMMON, F_COMMON, "mp").mean(axis=0)
             - np.log(cal_tang),
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
        print(f"Triptych saved (WDM div={rw0['divergences']}, MP div={rt0['divergences']})")

    # --- Figure 3: MSE, coverage, CI width, and runtime vs observations. ---
    # Keep every repeat so we can show the spread (median + interquartile band).
    div_total = 0
    for nt in args.nt:
        n_total = nt * NF
        cal_wdm, ref_wdm, ref_tang = _load_or_create_references(
            nt,
            args.freq_knots,
            args.reference_draws,
            args.reference_cache_dir,
            require_cache=args.require_reference_cache,
        )
        repeat_start = 0 if args.repeat_start is None else args.repeat_start
        output = (
            _shard_path(n_total, args.freq_knots)
            if args.repeat_start is None
            else _chunk_path(n_total, args.freq_knots, repeat_start, args.repeats)
        )
        rep = _load_checkpoint(
            output,
            n_total=n_total,
            freq_knots=args.freq_knots,
            repeat_start=repeat_start,
            repeats_target=args.repeats,
        )
        t0 = time.time()
        completed = len(rep[METRIC_KEYS[0]])
        for local_repeat in range(completed, args.repeats):
            r = repeat_start + local_repeat
            seed = 6000 + r
            data = simulate_ls2(n_total, rng=np.random.default_rng(seed))
            rw = rt = None
            try:
                rw, rt = _fit_both(data, nt, seed, args.freq_knots)
                div_total += rw["divergences"] + rt["divergences"]
                mw, cw, ww, nw = _metrics(
                    rw, cal_wdm, log_f0_common, "wdm", ref_wdm
                )
                mt, ct, wt, ntarget = _metrics(
                    rt, cal_tang, log_f0_common, "mp", ref_tang
                )
                rhat_w, neff_w = _diag_extrema(rw)
                rhat_t, neff_t = _diag_extrema(rt)
                rep["wm"].append(mw); rep["tm"].append(mt)
                rep["wc"].append(cw); rep["tc"].append(ct)
                rep["ww"].append(ww); rep["tw"].append(wt)
                rep["wn"].append(nw); rep["tn"].append(ntarget)
                rep["wt"].append(rw["nuts_runtime_s"]); rep["tt"].append(rt["nuts_runtime_s"])
                rep["wr"].append(rhat_w); rep["tr"].append(rhat_t)
                rep["we"].append(neff_w); rep["te"].append(neff_t)
                _atomic_savez(
                    output,
                    **_checkpoint_arrays(
                        rep,
                        n_total=n_total,
                        freq_knots=args.freq_knots,
                        repeat_start=repeat_start,
                        repeats_target=args.repeats,
                    ),
                )
                print(
                    f"n={n_total} repeat={r} checkpointed "
                    f"({local_repeat + 1}/{args.repeats})",
                    flush=True,
                )
            finally:
                del rw, rt, data
                jax.clear_caches()
                gc.collect()
        print(f"n={n_total:6d}  WDM mse={np.median(rep['wm']):.3f} "
              f"cov={np.mean(rep['wc']):.2f} ciw={np.median(rep['ww']):.2f} "
              f"t={np.median(rep['wt']):.1f}s rhat<={np.max(rep['wr']):.3f} "
              f"native={np.median(rep['wn']):.3f} neff>={np.min(rep['we']):.0f}  "
              f"MP mse={np.median(rep['tm']):.3f} cov={np.mean(rep['tc']):.2f} "
              f"ciw={np.median(rep['tw']):.2f} t={np.median(rep['tt']):.1f}s "
              f"native={np.median(rep['tn']):.3f} "
              f"rhat<={np.max(rep['tr']):.3f} neff>={np.min(rep['te']):.0f}  "
              f"({time.time()-t0:.0f}s)")
    print(f"total divergences: {div_total}  ({args.repeats} repeats/point)")

    if not args.skip_render:
        durations, raw = _load_shards(args.freq_knots)
        _render_metrics(durations, raw, args.freq_knots)
        _render_knot_sensitivity()
        print(f"Figure 3 saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
