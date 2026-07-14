"""MAP-first knot-allocation benchmark on the 30-day Ollie LISA data.

This deliberately keeps the experiment outside the inference API.  It compares
equal-size spline bases using a blocked time-bin holdout, then writes a compact
JSON summary.  The fitted band is the *effective* post-decimation band
(about 0.133 Hz), not 0.1 Hz, so the 0.12-Hz TDI null remains in the test.

Strategies:
  uniform       uniform time and physical-frequency coordinates
  current       current 1-D adaptive-time rule, uniform physical frequency
  hand_warp     current adaptive time plus the existing hand frequency warp
  pilot_time    2-D pilot curvature allocation in time only
  pilot_freq    2-D pilot curvature allocation in frequency only
  explicit_freq same pilot frequency knots, with roughness kept in physical f
  adaptive_2d   time/frequency curvature densities from a 2-D pilot MAP fit

Run a quick smoke test or the intended benchmark with, respectively,

    uv run python studies/ollie_tdi/knot_map_benchmark.py --maxiter 10
    uv run python studies/ollie_tdi/knot_map_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kurtosis

from tv_pspline_psd import PSplineConfig, wdm_analysis_coefficients
from tv_pspline_psd.model import power_floor
from tv_pspline_psd.splines import (
    create_adaptive_time_knots,
    create_bspline_basis,
    create_bspline_roughness_penalty,
    evaluate_bspline_basis,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "ollie_data" / "simulated_noise_30_days_L1_ext.h5"
RESULTS = REPO / "studies" / "results" / "ollie_tdi" / "knot_map_benchmark"

DT_RAW = 0.25
DECIMATE = 15
NT = 128
TRIM_TIME = 4
TRIM_LOW = 4
F_LO, F_BREAK = 1e-4, 0.02
N_HAND_LOG, N_HAND_LIN = 24, 70


@dataclass
class MapFit:
    W: np.ndarray
    log_scale: float
    objective: float
    success: bool
    nit: int
    runtime_s: float


def fft_decimate(x: np.ndarray, q: int) -> np.ndarray:
    n_new = x.size // q
    spec = np.fft.rfft(x[: n_new * q])
    return np.fft.irfft(spec[: n_new // 2 + 1], n=n_new) / q


def load_coefficients(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache.exists():
        z = np.load(cache)
        return z["coeffs"], z["time_grid"], z["freq_grid"]
    with h5py.File(DATA) as h:
        # Stored A2 is the same static orthonormal A combination used by the
        # full-band study and avoids reading X2 and Z2 separately.
        raw = h["tdis/A2"][:]
    data = fft_decimate(raw, DECIMATE)
    config = PSplineConfig(
        trim_time_bins=TRIM_TIME,
        trim_low_freq_channels=TRIM_LOW,
        trim_high_freq_channels=1,
        adaptive_time_knots=False,
    )
    coeffs, time_grid, freq_grid = wdm_analysis_coefficients(
        data, DT_RAW * DECIMATE, NT, config
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, coeffs=coeffs, time_grid=time_grid, freq_grid=freq_grid)
    return coeffs, time_grid, freq_grid


def normalize_coordinate(x: np.ndarray) -> np.ndarray:
    return (x - x[0]) / (x[-1] - x[0])


def hand_frequency_coordinate(freq: np.ndarray) -> np.ndarray:
    """The existing log-below-0.02-Hz / linear-above frequency warp."""
    f_hi = 1.0 / (2.0 * DT_RAW * DECIMATE)
    a = N_HAND_LOG / (N_HAND_LOG + N_HAND_LIN)
    low = a * np.log10(np.maximum(freq, F_LO) / F_LO) / np.log10(F_BREAK / F_LO)
    high = a + (1.0 - a) * (freq - F_BREAK) / (f_hi - F_BREAK)
    return normalize_coordinate(np.where(freq <= F_BREAK, low, high))


def coordinate_from_interior(x: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Piecewise-linear warp making supplied physical knots uniformly spaced."""
    anchors = np.r_[x[0], np.sort(interior), x[-1]]
    return np.interp(x, anchors, np.linspace(0.0, 1.0, anchors.size))


def block_frequency(power: np.ndarray, coord: np.ndarray, block: int):
    """Aggregate exact Whittle sufficient statistics within frequency blocks."""
    if block <= 1:
        return power, coord, np.ones(coord.size)
    n = coord.size
    starts = np.arange(0, n, block)
    summed = np.add.reduceat(power, starts, axis=1)
    counts = np.minimum(block, n - starts).astype(float)
    centers = np.array([coord[s:s + block].mean() for s in starts])
    return summed, centers, counts


def fit_map(
    power: np.ndarray,
    time_coord: np.ndarray,
    freq_coord: np.ndarray,
    *,
    n_time_knots: int,
    n_freq_knots: int,
    maxiter: int,
    freq_block: int,
    lambda_time: float,
    lambda_freq: float,
    interior_time: np.ndarray | None = None,
    interior_freq: np.ndarray | None = None,
) -> tuple[MapFit, np.ndarray, np.ndarray]:
    p_block, f_block, counts = block_frequency(power, freq_coord, freq_block)
    scale = float(np.median(power[power > 0]))
    p_block = p_block / scale
    Bt, kt = create_bspline_basis(
        time_coord, n_time_knots, degree=3, interior_knots=interior_time,
    )
    Bf, kf = create_bspline_basis(
        f_block, n_freq_knots, degree=3, interior_knots=interior_freq,
    )
    Pt = create_bspline_roughness_penalty(kt, degree=3, derivative_order=2)
    Pf = create_bspline_roughness_penalty(kf, degree=3, derivative_order=2)

    # A separable least-squares initialization is cheap and scale-free.
    target = np.log(p_block / counts[None, :] + power_floor(p_block))
    left = np.linalg.lstsq(Bt, target, rcond=1e-8)[0]
    W0 = np.linalg.lstsq(Bf, left.T, rcond=1e-8)[0].T
    shape = W0.shape

    def objective_gradient(flat: np.ndarray):
        W = flat.reshape(shape)
        eta = Bt @ W @ Bf.T
        inv = np.exp(np.clip(-eta, -60.0, 60.0))
        residual = 0.5 * (counts[None, :] - p_block * inv)
        penalty = 0.5 * (
            lambda_time * np.sum(W * (Pt @ W))
            + lambda_freq * np.sum(W * (W @ Pf))
        )
        value = 0.5 * np.sum(counts[None, :] * eta + p_block * inv) + penalty
        grad = (
            Bt.T @ residual @ Bf
            + lambda_time * Pt @ W
            + lambda_freq * W @ Pf
        )
        return float(value), grad.ravel()

    started = time.perf_counter()
    opt = minimize(
        objective_gradient, W0.ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-5, "maxls": 30},
    )
    fit = MapFit(
        W=opt.x.reshape(shape), log_scale=float(np.log(scale)),
        objective=float(opt.fun), success=bool(opt.success), nit=int(opt.nit),
        runtime_s=float(time.perf_counter() - started),
    )
    return fit, kt, kf


def surface(fit: MapFit, time_coord: np.ndarray, freq_coord: np.ndarray,
            knots_time: np.ndarray, knots_freq: np.ndarray) -> np.ndarray:
    Bt = evaluate_bspline_basis(time_coord, knots_time, degree=3)
    Bf = evaluate_bspline_basis(freq_coord, knots_freq, degree=3)
    return fit.log_scale + Bt @ fit.W @ Bf.T


def rms_density(values: np.ndarray, axis: int) -> np.ndarray:
    sq = values * values
    cap = np.percentile(sq, 99.0)
    score = np.sqrt(np.mean(np.minimum(sq, cap), axis=axis))
    return score / max(float(np.mean(score)), 1e-12)


def density_knots(x: np.ndarray, density: np.ndarray, n: int) -> np.ndarray:
    density = 0.2 + np.sqrt(np.maximum(density, 0.0))
    dx = np.diff(x)
    cdf = np.r_[0.0, np.cumsum(0.5 * (density[:-1] + density[1:]) * dx)]
    cdf /= cdf[-1]
    adaptive = np.interp(np.linspace(0, 1, n + 2)[1:-1], cdf, x)
    # A small uniform mixture prevents nearly coincident knots and unstable bases.
    uniform = np.linspace(x[0], x[-1], n + 2)[1:-1]
    return 0.9 * adaptive + 0.1 * uniform


def adaptive_coordinates(
    power_train: np.ndarray,
    time_train: np.ndarray,
    time_all: np.ndarray,
    freq: np.ndarray,
    *,
    n_time_knots: int,
    n_freq_knots: int,
    maxiter: int,
    freq_block: int,
    lambda_time: float,
    lambda_freq: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit a 2-D pilot and allocate both marginal knot densities by curvature."""
    t0 = normalize_coordinate(time_all)
    tt = np.interp(time_train, time_all, t0)
    fh = hand_frequency_coordinate(freq)
    pilot, kt, kf = fit_map(
        power_train, tt, fh, n_time_knots=n_time_knots,
        n_freq_knots=n_freq_knots, maxiter=maxiter, freq_block=freq_block,
        lambda_time=lambda_time, lambda_freq=lambda_freq,
    )
    eta = surface(pilot, t0, fh, kt, kf)
    dtt = np.gradient(np.gradient(eta, t0, axis=0), t0, axis=0)
    dff = np.gradient(np.gradient(eta, freq, axis=1), freq, axis=1)
    dtf = np.gradient(np.gradient(eta, t0, axis=0), freq, axis=1)
    rho_t = rms_density(dtt, axis=1) + 0.5 * rms_density(dtf, axis=1)
    rho_f = rms_density(dff, axis=0) + 0.5 * rms_density(dtf, axis=0)
    tk = density_knots(time_all, rho_t, n_time_knots)
    fk = density_knots(freq, rho_f, n_freq_knots)
    return coordinate_from_interior(time_all, tk), coordinate_from_interior(freq, fk), {
        "pilot": {k: v for k, v in asdict(pilot).items() if k != "W"},
        "time_knots": tk.tolist(), "freq_knots": fk.tolist(),
    }


def score(coeffs: np.ndarray, log_s: np.ndarray, freq: np.ndarray) -> dict[str, float]:
    power = coeffs**2
    floor = power_floor(power)
    ratio = power * np.exp(np.clip(-log_s, -100.0, 100.0))
    saturated = np.log(power + floor) + 1.0
    excess = 0.5 * (log_s + ratio - saturated)
    z = coeffs * np.exp(np.clip(-0.5 * log_s, -100.0, 100.0))
    nulls = np.arange(0.03, freq[-1] + 0.015, 0.03)
    null_region = np.min(np.abs(freq[:, None] - nulls[None, :]), axis=1) < 0.002
    return {
        "mean_excess_whittle_deviance": float(np.mean(excess)),
        "null_excess_whittle_deviance": float(np.mean(excess[:, null_region])),
        "non_null_excess_whittle_deviance": float(np.mean(excess[:, ~null_region])),
        "mean_z2": float(np.mean(ratio)),
        "time_z2_rmse": float(np.sqrt(np.mean((ratio.mean(axis=1) - 1.0) ** 2))),
        "freq_z2_rmse": float(np.sqrt(np.mean((ratio.mean(axis=0) - 1.0) ** 2))),
        "z_excess_kurtosis": float(kurtosis(z.ravel(), fisher=True, bias=False)),
    }


def plot_summary(
    output: dict[str, object], power: np.ndarray, freq: np.ndarray, path: Path,
) -> None:
    """Plot the mean spectrum, frequency-knot rugs, and held-out null score."""
    plt.style.use("default")
    strategies = output["strategies"]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.0), constrained_layout=True)
    ax = axes[0]
    ax.loglog(freq, np.mean(power, axis=0), color="0.25", lw=0.7)
    ax.set_ylabel(r"mean WDM power")
    ax.set_title("30-day A2 spectrum and frequency-knot allocation")
    uniform = np.linspace(freq[0], freq[-1], output["arguments"]["freq_knots"] + 2)[1:-1]
    rugs: list[tuple[str, np.ndarray, str]] = [("uniform/current", uniform, "0.55")]
    if "hand_warp" in strategies:
        u = np.linspace(0, 1, output["arguments"]["freq_knots"] + 2)[1:-1]
        rugs.append(("hand warp", np.interp(u, hand_frequency_coordinate(freq), freq), "tab:orange"))
    pilot = strategies.get("pilot_freq") or strategies.get("adaptive_2d")
    if pilot and pilot.get("freq_knots"):
        rugs.append(("2-D pilot", np.asarray(pilot["freq_knots"]), "tab:blue"))
    ymin, ymax = ax.get_ylim()
    levels = np.geomspace(ymax / 1.8, ymax / 7.0, len(rugs))
    for level, (label, knots, color) in zip(levels, rugs):
        ax.vlines(knots, level / 1.12, level * 1.12, color=color, lw=0.55)
        ax.text(freq[0] * 1.18, level, label, color=color, fontsize=7,
                ha="left", va="center")
    ax.set_ylim(ymin, ymax * 1.2)

    shown = [
        name for name in (
            "uniform", "current", "hand_warp", "pilot_time", "explicit_freq",
            "pilot_freq", "adaptive_2d",
        ) if name in strategies
    ]
    values = [strategies[name]["heldout"]["null_excess_whittle_deviance"] for name in shown]
    colors = ["tab:blue" if "freq" in name or name == "adaptive_2d" else "0.55" for name in shown]
    axes[1].bar(np.arange(len(shown)), values, color=colors)
    axes[1].set_xticks(np.arange(len(shown)), shown, rotation=25, ha="right")
    axes[1].set_ylabel("held-out null-region deviance")
    axes[1].set_ylim(0, max(values) * 1.15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time-knots", type=int, default=16)
    p.add_argument("--freq-knots", type=int, default=94)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument(
        "--freq-block", type=int, default=1,
        help="frequency sufficient-statistic block size; 1 keeps the exact grid",
    )
    p.add_argument("--lambda-time", type=float, default=2.0)
    p.add_argument("--lambda-freq", type=float, default=2.0)
    p.add_argument(
        "--strategies", nargs="+",
        default=[
            "uniform", "current", "hand_warp", "pilot_time", "pilot_freq",
            "explicit_freq", "adaptive_2d",
        ],
    )
    p.add_argument("--output", type=Path, default=RESULTS / "summary.json")
    args = p.parse_args()

    coeffs, time_grid, freq_grid = load_coefficients(RESULTS / "wdm_coeffs.npz")
    power = coeffs**2
    index = np.arange(time_grid.size)
    holdout = ((index % 10) == 4) | ((index % 10) == 5)
    holdout[[0, -1]] = False
    train = ~holdout
    t_uniform = normalize_coordinate(time_grid)
    current_knots = create_adaptive_time_knots(
        time_grid[train], np.mean(np.log(power[train] + power_floor(power[train])), axis=1),
        n_interior_knots=args.time_knots, smoothing_sigma=1.0, variation_floor=0.25,
    )
    current_t = coordinate_from_interior(time_grid, current_knots)
    coords: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {
        "uniform": (t_uniform, normalize_coordinate(freq_grid), {}),
        "current": (current_t, normalize_coordinate(freq_grid), {"time_knots": current_knots.tolist()}),
        "hand_warp": (current_t, hand_frequency_coordinate(freq_grid), {"time_knots": current_knots.tolist()}),
    }
    explicit: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    pilot_strategies = {"pilot_time", "pilot_freq", "explicit_freq", "adaptive_2d"}
    if pilot_strategies.intersection(args.strategies):
        ta, fa, meta = adaptive_coordinates(
            power[train], time_grid[train], time_grid, freq_grid,
            n_time_knots=args.time_knots, n_freq_knots=args.freq_knots,
            maxiter=args.maxiter, freq_block=args.freq_block,
            lambda_time=args.lambda_time, lambda_freq=args.lambda_freq,
        )
        coords["pilot_time"] = (
            ta, normalize_coordinate(freq_grid),
            {"time_knots": meta["time_knots"], "pilot": meta["pilot"]},
        )
        coords["pilot_freq"] = (
            t_uniform, fa,
            {"freq_knots": meta["freq_knots"], "pilot": meta["pilot"]},
        )
        freq_uniform = normalize_coordinate(freq_grid)
        explicit_freq = np.interp(meta["freq_knots"], freq_grid, freq_uniform)
        coords["explicit_freq"] = (
            t_uniform, freq_uniform,
            {"freq_knots": meta["freq_knots"], "pilot": meta["pilot"]},
        )
        explicit["explicit_freq"] = (None, explicit_freq)
        coords["adaptive_2d"] = (ta, fa, meta)

    output: dict[str, object] = {
        "dataset": str(DATA), "channel": "A2", "dt_raw_s": DT_RAW,
        "decimate": DECIMATE, "dt_s": DT_RAW * DECIMATE, "nt": NT,
        "grid_shape": list(coeffs.shape), "frequency_band_hz": [float(freq_grid[0]), float(freq_grid[-1])],
        "note": "Full effective 0.133-Hz band retained, including the 0.12-Hz null.",
        "holdout_rows": np.flatnonzero(holdout).tolist(), "arguments": vars(args) | {"output": str(args.output)},
        "strategies": {},
    }
    for name in args.strategies:
        tc, fc, meta = coords[name]
        fit, kt, kf = fit_map(
            power[train], tc[train], fc, n_time_knots=args.time_knots,
            n_freq_knots=args.freq_knots, maxiter=args.maxiter,
            freq_block=args.freq_block, lambda_time=args.lambda_time,
            lambda_freq=args.lambda_freq,
            interior_time=explicit.get(name, (None, None))[0],
            interior_freq=explicit.get(name, (None, None))[1],
        )
        log_s = surface(fit, tc[holdout], fc, kt, kf)
        fit_info = {k: v for k, v in asdict(fit).items() if k != "W"}
        result = {"map": fit_info, "heldout": score(coeffs[holdout], log_s, freq_grid), **meta}
        output["strategies"][name] = result
        print(name, json.dumps(result["heldout"], sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    plot_summary(output, power, freq_grid, args.output.with_suffix(".png"))
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
