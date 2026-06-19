"""Convergence diagnostics for the WDM log-P-spline fit."""

from __future__ import annotations

import numpy as np
from numpyro.diagnostics import summary


def summarize_mcmc_diagnostics(results: dict[str, object]) -> dict[str, object]:
    """Compact NUTS diagnostics: divergences, smoothing precisions, latent pixels.

    The ``phi_time`` / ``phi_freq`` sites are sampled as ``log phi``; their
    ``r_hat`` and ``n_eff`` are reported on that (sampled) scale, while the
    reported ``mean`` is exponentiated back to the natural precision scale.
    """
    mcmc = results["mcmc"]
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = summary(grouped, group_by_chain=True)
    divergences = int(
        np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum()
    )

    psd_mean = np.asarray(results["psd_mean"])
    n_time, n_freq = psd_mean.shape
    probes = [
        ("center", n_time // 2, n_freq // 2),
        ("low_freq", n_time // 2, max(1, n_freq // 5)),
        ("high_freq", n_time // 2, min(n_freq - 2, (4 * n_freq) // 5)),
    ]
    latent = {}
    if "log_psd" in diag:  # absent when the surface samples are not stored
        for label, i, j in probes:
            site = diag["log_psd"]
            latent[label] = {
                "index": (i, j),
                "mean": float(site["mean"][i, j]),
                "n_eff": float(site["n_eff"][i, j]),
                "r_hat": float(site["r_hat"][i, j]),
            }

    def _phi(name: str) -> dict[str, float]:
        return {
            "mean": float(np.exp(diag[name]["mean"])),
            "n_eff": float(diag[name]["n_eff"]),
            "r_hat": float(diag[name]["r_hat"]),
        }

    return {
        "num_chains": int(grouped["phi_time"].shape[0]),
        "divergences": divergences,
        "phi_time": _phi("phi_time"),
        "phi_freq": _phi("phi_freq"),
        "latent_log_psd": latent,
    }
