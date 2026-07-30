"""Controlled diagonal-quasi-likelihood study using a Cornish Gram factor.

The simulator uses the independent real cosine/sine basis described as an
alternative to Hermitian-symmetrising Cornish's complex Fourier construction.
For a prescribed positive dynamic spectrum ``S(f,t)=s(f,t)**2``, the resulting
time series has covariance ``A A.T`` by construction.

Both front ends see the same realizations.  Their actual finite-resolution
marginal-power targets are computed exactly from the Gram factor:

* WDM: ``E[w_nm**2]`` on the trimmed WDM grid;
* moving periodogram: ``E[abs(d_i)**2]`` at the retained zigzag ordinates.

The same factor gives the exact largest local coefficient correlation on each
representation's native neighbourhood.  The campaign reports log-scale
bias/MSE and pointwise 90% credible-interval coverage against the exact targets,
alongside an independent-WDM-coefficient control and convergence-gated summaries.

Example:
    python studies/cornish_gram_factor_study.py --repeats 35 \
        --warmup 500 --samples 500 --num-chains 2 --run-name baseline_n35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from wdm_transform import TimeSeries

from tv_pspline_psd import (
    PSplineConfig,
    fit_log_pspline_surface,
    run_tang_dynamic_whittle_mcmc,
    run_wdm_psd_mcmc,
    tang_moving_periodogram,
    wdm_analysis_coefficients,
)
from tv_pspline_psd.splines import evaluate_bspline_basis

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "cornish_gram_factor"


def tilted_bump_spectrum(
    n_total: int,
    dt: float,
    *,
    amplitude: float = 0.5,
    correlation: float = 0.5,
    center_time: float = 0.55,
    center_freq_fraction: float = 0.38,
    sigma_time: float = 0.125,
    sigma_freq_fraction: float = 0.065,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cornish-style tilted Gaussian bump on a smooth positive background."""
    time = (np.arange(n_total) + 0.5) / n_total
    freq = np.fft.rfftfreq(n_total, dt)
    nyquist = 1.0 / (2.0 * dt)
    f0 = center_freq_fraction * nyquist
    sigma_f = sigma_freq_fraction * nyquist
    # A positive, gently sloped background avoids a singular DC component while
    # retaining enough frequency gradient to induce realistic WDM leakage.
    background = 0.35 + (freq / max(f0, np.finfo(float).eps)) ** 2
    x = (freq[None, :] - f0) / sigma_f
    y = (time[:, None] - center_time) / sigma_time
    qform = (x**2 - 2.0 * correlation * x * y + y**2) / (1.0 - correlation**2)
    spectrum = background[None, :] * (1.0 + amplitude * np.exp(-0.5 * qform))
    return time, freq, spectrum


def real_gram_factor(spectrum: np.ndarray) -> np.ndarray:
    """Return a real covariance square root for a one-sided dynamic spectrum.

    With ``eta ~ N(0,I)``, ``x = factor @ eta`` is a real Gaussian realization.
    For a flat unit spectrum, every time sample has exactly unit variance.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n_total, n_rfft = spectrum.shape
    if n_rfft != n_total // 2 + 1 or n_total % 2:
        raise ValueError("spectrum must have shape (even N, N//2+1)")
    if not np.isfinite(spectrum).all() or np.any(spectrum <= 0.0):
        raise ValueError("dynamic spectrum must be finite and strictly positive")
    amplitude = np.sqrt(spectrum)
    n = np.arange(n_total)[:, None]
    k = np.arange(1, n_total // 2)[None, :]
    phase = 2.0 * np.pi * n * k / n_total
    factor = np.empty((n_total, n_total), dtype=float)
    factor[:, 0] = amplitude[:, 0] / np.sqrt(n_total)
    factor[:, 1] = amplitude[:, -1] * ((-1.0) ** n[:, 0]) / np.sqrt(n_total)
    stop = 2 + k.size
    scale = np.sqrt(2.0 / n_total) * amplitude[:, 1:-1]
    factor[:, 2:stop] = scale * np.cos(phase)
    factor[:, stop:] = -scale * np.sin(phase)
    return factor


def draw_realizations(
    factor: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw columns of time series from a fixed covariance square root."""
    return factor @ rng.standard_normal((factor.shape[1], n_draws))


def exact_wdm_target(
    factor: np.ndarray,
    *,
    dt: float,
    nt: int,
    config: PSplineConfig,
) -> tuple[np.ndarray, float]:
    """Return exact WDM marginal variances and maximum 3x17 correlation."""
    target = None
    covariance = {}
    # Treat Gram-factor columns as independent channels.  Summing their squared
    # WDM responses gives the exact marginal variance without forming A A.T.
    batch_size = 64
    for start in range(0, factor.shape[1], batch_size):
        stop = min(start + batch_size, factor.shape[1])
        coeff = np.asarray(
            TimeSeries(factor[:, start:stop].T, dt=dt).to_wdm(nt=nt).coeffs
        )
        coeff = coeff[
            :,
            config.trim_time_bins : coeff.shape[1] - config.trim_time_bins,
            config.trim_low_freq_channels : coeff.shape[2]
            - config.trim_high_freq_channels,
        ]
        if target is None:
            target = np.zeros(coeff.shape[1:], dtype=float)
            for dn in range(-1, 2):
                for dm in range(-8, 9):
                    if dn != 0 or dm != 0:
                        n_size = coeff.shape[1] - abs(dn)
                        m_size = coeff.shape[2] - abs(dm)
                        covariance[(dn, dm)] = np.zeros((n_size, m_size), dtype=float)
        target += np.sum(coeff**2, axis=0)
        for (dn, dm), accumulated in covariance.items():
            n0 = slice(max(0, -dn), coeff.shape[1] - max(0, dn))
            n1 = slice(max(0, dn), coeff.shape[1] - max(0, -dn))
            m0 = slice(max(0, -dm), coeff.shape[2] - max(0, dm))
            m1 = slice(max(0, dm), coeff.shape[2] - max(0, -dm))
            accumulated += np.sum(coeff[:, n0, m0] * coeff[:, n1, m1], axis=0)
    if target is None:
        raise ValueError("Gram factor must contain at least one column")
    max_correlation = 0.0
    for (dn, dm), cross in covariance.items():
        n0 = slice(max(0, -dn), target.shape[0] - max(0, dn))
        n1 = slice(max(0, dn), target.shape[0] - max(0, -dn))
        m0 = slice(max(0, -dm), target.shape[1] - max(0, dm))
        m1 = slice(max(0, dm), target.shape[1] - max(0, -dm))
        corr = cross / np.sqrt(target[n0, m0] * target[n1, m1])
        max_correlation = max(max_correlation, float(np.max(np.abs(corr))))
    return target, max_correlation


def exact_moving_target(
    factor: np.ndarray, *, m: int, thin: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return exact moving variances, coordinates, and maximum local correlation."""
    n_total = factor.shape[0]
    template = tang_moving_periodogram(np.zeros(n_total), m=m, thin=thin)
    n_blocks = (n_total - 2 * m) // (thin * m)
    window_starts = (
        thin * m * np.arange(n_blocks)[:, None] + np.arange(m)[None, :]
    ).reshape(-1)
    freq_index = np.tile(np.arange(m), n_blocks)
    nu = np.arange(2 * m + 1)
    lam = 2.0 * np.arange(1, m + 1) / (2 * m + 1)
    phase_by_freq = np.exp(-1j * np.pi * np.outer(nu, lam)).T
    norm = np.sqrt(2.0 * np.pi * (2 * m + 1))
    responses = np.empty((window_starts.size, factor.shape[1]), dtype=np.complex128)
    # A bounded chunk avoids materialising the full ordinates x window x N
    # response tensor (about 8 GiB for the default pilot).
    chunk_points = 16
    row_offsets = np.arange(2 * m + 1)
    for start in range(0, window_starts.size, chunk_points):
        stop = min(start + chunk_points, window_starts.size)
        rows = window_starts[start:stop, None] + row_offsets[None, :]
        selected_factor = factor[rows]
        weights = phase_by_freq[freq_index[start:stop]] / norm
        responses[start:stop] = np.einsum(
            "pl,pln->pn", weights, selected_factor, optimize=True
        )
    target = np.sum(np.abs(responses) ** 2, axis=1)
    response_grid = responses.reshape(n_blocks, m, factor.shape[1])
    target_grid = target.reshape(n_blocks, m)
    max_correlation = 0.0
    for db in range(-1, 2):
        for dj in range(-8, 9):
            if db == 0 and dj == 0:
                continue
            if abs(db) >= n_blocks or abs(dj) >= m:
                continue
            b0 = slice(max(0, -db), n_blocks - max(0, db))
            b1 = slice(max(0, db), n_blocks - max(0, -db))
            j0 = slice(max(0, -dj), m - max(0, dj))
            j1 = slice(max(0, dj), m - max(0, -dj))
            cross = np.sum(
                np.conj(response_grid[b0, j0]) * response_grid[b1, j1], axis=-1
            )
            denom = np.sqrt(target_grid[b0, j0] * target_grid[b1, j1])
            max_correlation = max(max_correlation, float(np.max(np.abs(cross) / denom)))
    return target, template["u"], template["omega"], max_correlation


def exact_targets_and_dependence(
    factor: np.ndarray,
    *,
    dt: float,
    nt: int,
    m: int,
    thin: int,
    config: PSplineConfig,
) -> dict[str, np.ndarray | float]:
    """Exact marginal targets and dependence from the known Gram factor."""
    wdm_target, wdm_corr = exact_wdm_target(factor, dt=dt, nt=nt, config=config)
    mp_target, mp_u, mp_omega, mp_corr = exact_moving_target(factor, m=m, thin=thin)
    return {
        "wdm_target": wdm_target,
        "mp_target": mp_target,
        "mp_u": mp_u,
        "mp_omega": mp_omega,
        "wdm_max_local_corr": wdm_corr,
        "mp_max_local_corr": mp_corr,
    }


def moving_log_surface_at_ordinates(result: dict[str, object]) -> np.ndarray:
    """Evaluate every posterior draw at the retained native ordinate locations."""
    ordinates = result["ordinates"]
    config = result["config"]
    bt = (
        evaluate_bspline_basis(
            ordinates["u"], result["knots_time"], degree=config.degree_time
        )
        @ result["whitened"]["U_time"]
    )
    bf = (
        evaluate_bspline_basis(
            ordinates["omega"] / np.pi,
            result["knots_freq"],
            degree=config.degree_freq,
        )
        @ result["whitened"]["U_freq"]
    )
    return np.einsum(
        "ia,nab,ib->ni", bt, result["eig_coeff_samples"], bf, optimize=True
    )


def surface_metrics(
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    delta = np.log(estimate) - np.log(target)
    log_width = np.log(upper) - np.log(lower)
    return {
        "mean_log_bias": float(np.mean(delta)),
        "log_mse": float(np.mean(delta**2)),
        "coverage_90": float(np.mean((lower <= target) & (target <= upper))),
        "mean_log_interval_width_90": float(np.mean(log_width)),
    }


def sampler_metrics(result: dict[str, object]) -> dict[str, float | int | None]:
    """Rank-normalised R-hat and bulk ESS for the smoothing precisions."""
    import arviz as az

    grouped = result["mcmc"].get_samples(group_by_chain=True)

    def scalar(dataset) -> float:
        return float(dataset.to_array().values.reshape(-1)[0])

    rhats = np.asarray(
        [
            scalar(az.rhat({name: grouped[name]}, method="rank"))
            for name in ("phi_time", "phi_freq")
        ]
    )
    esses = np.asarray(
        [
            scalar(az.ess({name: grouped[name]}, method="bulk"))
            for name in ("phi_time", "phi_freq")
        ]
    )
    finite_rhat = rhats[np.isfinite(rhats)]
    finite_ess = esses[np.isfinite(esses)]
    return {
        "divergences": int(result["divergences"]),
        "max_rhat": float(finite_rhat.max()) if finite_rhat.size else None,
        "min_ess": float(finite_ess.min()) if finite_ess.size else None,
        "nuts_runtime_s": float(result["nuts_runtime_s"]),
    }


def repeat_passes_convergence(row: dict[str, object]) -> bool:
    """Require every matched fit to pass the manuscript's convergence gate."""
    for name in ("wdm", "moving_periodogram", "independent_wdm_control"):
        metrics = row[name]
        if metrics["divergences"] != 0:
            return False
        if metrics["max_rhat"] is None or metrics["max_rhat"] >= 1.01:
            return False
        if metrics["min_ess"] is None or metrics["min_ess"] < 300:
            return False
    return True


def summarize_repeats(per_repeat: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate location and spread while preserving realization-level rows."""
    summary = {}
    methods = ("wdm", "moving_periodogram", "independent_wdm_control")
    metrics = (
        "mean_log_bias",
        "log_mse",
        "coverage_90",
        "mean_log_interval_width_90",
        "divergences",
        "max_rhat",
        "min_ess",
        "nuts_runtime_s",
    )
    for name in methods:
        summary[name] = {}
        for metric in metrics:
            values = np.asarray(
                [
                    row[name][metric]
                    for row in per_repeat
                    if row[name].get(metric) is not None
                ],
                dtype=float,
            )
            summary[name][metric] = (
                {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
                if values.size
                else None
            )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    n_total = args.nt * args.nf
    _, freq, spectrum = tilted_bump_spectrum(
        n_total,
        args.dt,
        amplitude=args.amplitude,
        correlation=args.tilt_correlation,
        sigma_time=args.sigma_time,
        sigma_freq_fraction=args.sigma_freq_fraction,
    )
    factor = real_gram_factor(spectrum)
    config = PSplineConfig(
        n_interior_knots_time=args.time_knots,
        n_interior_knots_freq=args.freq_knots,
        trim_time_bins=1,
        trim_low_freq_channels=1,
        trim_high_freq_channels=1,
        centered=args.centered,
    )
    references = exact_targets_and_dependence(
        factor,
        dt=args.dt,
        nt=args.nt,
        m=args.m,
        thin=args.thin,
        config=config,
    )

    _, wdm_time_grid, wdm_freq_grid = wdm_analysis_coefficients(
        np.zeros(n_total), args.dt, args.nt, config
    )
    run_dir = args.outdir / args.run_name
    repeat_dir = run_dir / "repeats"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    per_repeat = []
    for repeat in range(args.repeats):
        repeat_path = repeat_dir / f"repeat_{repeat:03d}.json"
        if args.resume and repeat_path.exists():
            per_repeat.append(json.loads(repeat_path.read_text()))
            print(f"[{repeat + 1}/{args.repeats}] resumed {repeat_path.name}")
            continue
        rng = np.random.default_rng(args.seed + 10_000 + repeat)
        data = draw_realizations(factor, 1, rng)[:, 0]
        wdm = run_wdm_psd_mcmc(
            data,
            dt=args.dt,
            nt=args.nt,
            config=config,
            n_warmup=args.warmup,
            n_samples=args.samples,
            num_chains=args.num_chains,
            random_seed=args.seed + 100 + repeat,
        )
        moving = run_tang_dynamic_whittle_mcmc(
            data,
            dt=args.dt,
            m=args.m,
            thin=args.thin,
            config=config,
            n_time_grid=args.nt - 2,
            n_warmup=args.warmup,
            n_samples=args.samples,
            num_chains=args.num_chains,
            random_seed=args.seed + 200 + repeat,
        )
        independent_coeff = np.sqrt(
            np.asarray(references["wdm_target"])
        ) * rng.standard_normal(np.asarray(references["wdm_target"]).shape)
        independent = fit_log_pspline_surface(
            independent_coeff[None, :, :],
            wdm_time_grid,
            wdm_freq_grid,
            config=config,
            n_warmup=args.warmup,
            n_samples=args.samples,
            num_chains=args.num_chains,
            random_seed=args.seed + 300 + repeat,
        )
        mp_log_draws = moving_log_surface_at_ordinates(moving)
        mp_estimate = np.exp(mp_log_draws.mean(axis=0))
        mp_lower, mp_upper = np.exp(np.percentile(mp_log_draws, [5.0, 95.0], axis=0))
        row = {
            "repeat": repeat,
            "wdm": surface_metrics(
                np.asarray(wdm["psd_mean"]),
                np.asarray(wdm["psd_lower"]),
                np.asarray(wdm["psd_upper"]),
                np.asarray(references["wdm_target"]),
            )
            | sampler_metrics(wdm),
            "moving_periodogram": surface_metrics(
                mp_estimate,
                mp_lower,
                mp_upper,
                np.asarray(references["mp_target"]),
            )
            | sampler_metrics(moving),
            "independent_wdm_control": surface_metrics(
                np.asarray(independent["psd_mean"]),
                np.asarray(independent["psd_lower"]),
                np.asarray(independent["psd_upper"]),
                np.asarray(references["wdm_target"]),
            )
            | sampler_metrics(independent),
        }
        repeat_path.write_text(json.dumps(row, indent=2))
        per_repeat.append(row)
        print(
            f"[{repeat + 1}/{args.repeats}] "
            f"WDM mse={per_repeat[-1]['wdm']['log_mse']:.4f} "
            f"cov={per_repeat[-1]['wdm']['coverage_90']:.3f}; "
            f"MP mse={per_repeat[-1]['moving_periodogram']['log_mse']:.4f} "
            f"cov={per_repeat[-1]['moving_periodogram']['coverage_90']:.3f}; "
            f"control mse={per_repeat[-1]['independent_wdm_control']['log_mse']:.4f} "
            f"cov={per_repeat[-1]['independent_wdm_control']['coverage_90']:.3f}"
        )

    valid_repeats = [row for row in per_repeat if repeat_passes_convergence(row)]
    return {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dynamic_spectrum": {
            "shape": list(spectrum.shape),
            "min": float(spectrum.min()),
            "max": float(spectrum.max()),
            "rfft_freq_min": float(freq.min()),
            "rfft_freq_max": float(freq.max()),
        },
        "dependence": {
            "wdm_max_local_correlation": references["wdm_max_local_corr"],
            "moving_max_local_correlation": references["mp_max_local_corr"],
        },
        "targets": {
            "method": "exact Gram-factor marginal variances",
            "dependence_method": "exact Gram-factor local covariance",
        },
        "convergence_gate": {
            "max_rhat_exclusive": 1.01,
            "min_bulk_ess_inclusive": 300,
            "required_divergences": 0,
            "attempted_repeats": len(per_repeat),
            "valid_matched_repeats": len(valid_repeats),
            "excluded_repeat_indices": [
                row["repeat"]
                for row in per_repeat
                if not repeat_passes_convergence(row)
            ],
        },
        "summary": summarize_repeats(valid_repeats),
        "summary_all_attempts": summarize_repeats(per_repeat),
        "per_repeat": per_repeat,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nt", type=int, default=64)
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--time-knots", type=int, default=8)
    parser.add_argument("--freq-knots", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--tilt-correlation", type=float, default=0.5)
    parser.add_argument("--sigma-time", type=float, default=0.125)
    parser.add_argument("--sigma-freq-fraction", type=float, default=0.065)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument(
        "--centered", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=260713168)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nt * args.nf % 2:
        raise ValueError("nt*nf must be even")
    args.outdir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    out = args.outdir / args.run_name / "summary.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["dependence"], indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
