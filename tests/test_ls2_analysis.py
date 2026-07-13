from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tv_pspline_psd import (
    evaluate_dense_posterior_mean,
    mse_log_psd,
    run_stft_mcmc,
    run_wdm_psd_mcmc,
    save_figure,
    stft_white_noise_calibration,
)
from tv_pspline_psd.datasets import true_psd_ls2, wdm_white_noise_calibration


def _save_psd_plot(
    results: dict[str, object],
    calibration: np.ndarray,
    dt: float,
    mse_log: float,
    path: Path,
    title: str,
) -> Path:
    dense_results = evaluate_dense_posterior_mean(
        results,
        n_time_dense=160,
        n_freq_dense=160,
    )
    dense_time = np.asarray(dense_results["time_grid"])
    dense_freq = np.asarray(dense_results["freq_grid"])
    dense_calibration = np.interp(
        dense_freq,
        np.asarray(results["freq_grid"]),
        calibration,
    )
    true_log_psd = np.log(
        true_psd_ls2(dense_time, dense_freq, dt) * dense_calibration[None, :] + 1e-12
    )
    fitted_log_psd = np.log(np.asarray(dense_results["psd_mean"]) + 1e-12)
    vmin = min(true_log_psd.min(), fitted_log_psd.min())
    vmax = max(true_log_psd.max(), fitted_log_psd.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    for ax, field, panel_title in [
        (axes[0], true_log_psd, "True log PSD"),
        (axes[1], fitted_log_psd, f"Fitted log PSD (MSE={mse_log:.3f})"),
    ]:
        mesh = ax.pcolormesh(
            dense_time,
            dense_freq,
            field.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Rescaled time")
        fig.colorbar(mesh, ax=ax, label="log PSD")
    axes[0].set_ylabel("Frequency [Hz]")
    fig.suptitle(title)
    return save_figure(fig, path)


def test_ls2_smoke_runs_wdm_and_moving_periodogram(
    ls2_smoke_data: np.ndarray,
    smoke_config,
    plot_outdir,
) -> None:
    dt = 0.1
    nt = 32
    nperseg = 32

    wdm_results = run_wdm_psd_mcmc(
        ls2_smoke_data,
        dt=dt,
        nt=nt,
        config=smoke_config,
        n_warmup=12,
        n_samples=12,
        num_chains=1,
        random_seed=0,
    )
    stft_results = run_stft_mcmc(
        ls2_smoke_data,
        dt=dt,
        nperseg=nperseg,
        config=smoke_config,
        n_warmup=12,
        n_samples=12,
        num_chains=1,
        random_seed=0,
    )

    wdm_reference = true_psd_ls2(
        np.asarray(wdm_results["time_grid"]),
        np.asarray(wdm_results["freq_grid"]),
        dt,
    ) * wdm_white_noise_calibration(
        len(ls2_smoke_data),
        dt,
        nt,
        smoke_config,
        n_draws=32,
        seed=0,
    )[None, :]
    stft_reference = true_psd_ls2(
        np.asarray(stft_results["time_grid"]),
        np.asarray(stft_results["freq_grid"]),
        dt,
    ) * stft_white_noise_calibration(
        len(ls2_smoke_data),
        dt,
        nperseg,
        n_draws=32,
        seed=0,
    )[np.asarray(stft_results["keep_freq"]),][None, :]
    wdm_log_mse = mse_log_psd(wdm_reference, np.asarray(wdm_results["psd_mean"]))
    stft_log_mse = mse_log_psd(stft_reference, np.asarray(stft_results["psd_mean"]))
    wdm_raw_log_mse = mse_log_psd(wdm_reference, np.asarray(wdm_results["power"]))
    stft_raw_log_mse = mse_log_psd(
        stft_reference,
        np.asarray(stft_results["power"]) / np.asarray(stft_results["coeffs"]).shape[0],
    )

    assert wdm_results["psd_mean"].ndim == 2
    assert stft_results["psd_mean"].ndim == 2
    assert wdm_results["psd_mean"].shape == wdm_results["log_psd_mean"].shape
    assert stft_results["psd_mean"].shape == stft_results["log_psd_mean"].shape
    assert np.isfinite(wdm_results["psd_mean"]).all()
    assert np.isfinite(stft_results["psd_mean"]).all()
    assert wdm_log_mse < 0.90
    assert stft_log_mse < 0.90
    assert wdm_log_mse < wdm_raw_log_mse
    assert stft_log_mse < stft_raw_log_mse

    wdm_plot = _save_psd_plot(
        wdm_results,
        wdm_white_noise_calibration(
            len(ls2_smoke_data),
            dt,
            nt,
            smoke_config,
            n_draws=32,
            seed=0,
        ),
        dt,
        wdm_log_mse,
        Path(plot_outdir) / "ls2_wdm_psd.png",
        "LS2 WDM PSD",
    )
    stft_plot = _save_psd_plot(
        stft_results,
        stft_white_noise_calibration(
            len(ls2_smoke_data),
            dt,
            nperseg,
            n_draws=32,
            seed=0,
        )[np.asarray(stft_results["keep_freq"])],
        dt,
        stft_log_mse,
        Path(plot_outdir) / "ls2_stft_psd.png",
        "LS2 Moving-Periodogram PSD",
    )

    assert wdm_plot.exists()
    assert stft_plot.exists()
