"""Generate the LISA confusion-noise figures used in the manuscript.

Saves into ``notes/figures/``. Requires the project to be installed
(``uv pip install -e .`` / ``uv sync``), so no path manipulation is needed.

    python notes/scripts/make_lisa_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    LISANoiseConfig,
    galactic_modulation,
    monte_carlo_reference,
    normalization_constant,
    simulate_tv_lisa_noise,
    true_psd_lisa,
    wdm_white_noise_calibration,
)
from tv_pspline_psd import (
    PSplineConfig,
    interval_coverage,
    mse_log_psd,
    plot_channel_slice,
    plot_surface_comparison,
    relative_surface_error,
    run_wdm_psd_mcmc,
    save_figure,
    summarize_mcmc_diagnostics,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

DT, NT, N_TOTAL = 167.0, 24, 768
LISA = LISANoiseConfig(tobs_key="1yr", n_modulation_cycles=3.0)
PSPLINE = PSplineConfig(
    n_interior_knots_time=8, n_interior_knots_freq=10, trim_low_freq_channels=2
)


def _true_surface_illustration(reference: np.ndarray, time_grid: np.ndarray,
                               freq_grid: np.ndarray) -> None:
    """Seasonal modulation envelope + analytic PSD surface, for the method section.

    The raw time series is not shown because the very red LISA spectrum makes it
    look like a low-frequency sinusoid; the modulation envelope makes the
    non-stationarity explicit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    u = np.linspace(0.0, 1.0, 400)
    axes[0].plot(u, galactic_modulation(u, LISA) ** 2, color="tab:blue", lw=2.0)
    axes[0].set_title(r"Galactic confusion modulation $m(u)^2$")
    axes[0].set_xlabel("Rescaled time $u$")
    axes[0].set_ylabel(r"$m(u)^2$  ($\langle m^2 \rangle = 1$)")
    axes[0].axhline(1.0, color="black", ls="--", lw=1.0)

    mesh = axes[1].pcolormesh(
        time_grid, freq_grid * 1e3, np.log(reference + 1e-12).T,
        shading="nearest", cmap="viridis",
    )
    axes[1].set_title(r"Local power $\mathbb{E}[w^2] = S(u, f)$")
    axes[1].set_xlabel("Rescaled time $u$")
    axes[1].set_ylabel("Frequency [mHz]")
    fig.colorbar(mesh, ax=axes[1], label="log local power")
    save_figure(fig, FIG_DIR / "lisa_true_surface.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    data, _ = simulate_tv_lisa_noise(N_TOTAL, dt=DT, rng=rng, config=LISA)

    res = run_wdm_psd_mcmc(
        data, dt=DT, nt=NT, config=PSPLINE,
        n_warmup=400, n_samples=400, num_chains=2, random_seed=21,
    )
    reference = monte_carlo_reference(
        lambda r: simulate_tv_lisa_noise(N_TOTAL, dt=DT, rng=r, config=LISA)[0],
        n_draws=200, n_total=N_TOTAL, dt=DT, nt=NT, config=PSPLINE, seed=321,
    )
    calibration = wdm_white_noise_calibration(N_TOTAL, DT, NT, PSPLINE)
    norm_ref = normalization_constant(N_TOTAL, DT, LISA)
    true = calibration[None, :] * true_psd_lisa(
        res["time_grid"], res["freq_grid"], LISA, norm_ref=norm_ref
    )

    diag = summarize_mcmc_diagnostics(res)
    print(f"LISA fit: divergences={diag['divergences']}  "
          f"MSE_log(ref)={mse_log_psd(reference, res['psd_mean']):.3f}  "
          f"MSE_log(true)={mse_log_psd(true, res['psd_mean']):.3f}  "
          f"rel_err={relative_surface_error(reference, res['psd_mean']):.3f}  "
          f"cov={interval_coverage(reference, res['psd_lower'], res['psd_upper']):.2f}")

    _true_surface_illustration(reference, res["time_grid"], res["freq_grid"])
    plot_surface_comparison(
        res, reference, freq_scale=1e3, freq_label="Frequency [mHz]",
        path=FIG_DIR / "lisa_surface_comparison.png",
    )
    cv = reference.std(axis=0) / np.maximum(reference.mean(axis=0), 1e-30)
    channel = int(np.argmax(cv))
    plot_channel_slice(
        res, reference, channel, true_psd=true,
        freq_scale=1e3, freq_label="Confusion", path=FIG_DIR / "lisa_modulation_channel.png",
    )
    print(f"Saved LISA figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
