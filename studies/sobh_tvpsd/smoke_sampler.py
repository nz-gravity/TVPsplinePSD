"""Wire-test the WDM-branch joint SOBH + TV-PSD sampler at tiny scale.

Injects the differentiable signal at a known theta plus mildly time-varying
Gaussian noise (matched units), runs run_joint_sobh_wdm_mcmc from an offset
start, and checks recovery of (Mc, tc, dL) and the noise surface.

    uv run python studies/sobh_tvpsd/smoke_sampler.py
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from datasets.sobh import SOBHParams  # noqa: E402
from tv_pspline_psd import (  # noqa: E402
    PSplineConfig,
    build_sobh_wdm_grid,
    make_sobh_wdm_signal_fn,
    run_joint_sobh_wdm_mcmc,
)


def main() -> None:
    dt, nt, n = 4.0, 16, 2**13
    params = SOBHParams()
    params.tc = 0.6 * n * dt
    cfg = PSplineConfig()

    grid = build_sobh_wdm_grid(n, dt, nt, params, cfg)
    signal_fn = make_sobh_wdm_signal_fn(grid)
    theta_true = np.array([params.chirp_mass, params.tc, np.log(params.distance)])
    signal = np.asarray(signal_fn(jnp.asarray(theta_true)))

    # Mildly time-varying true noise surface, matched to the signal scale.
    rng = np.random.default_rng(0)
    sigma = 0.7 * np.sqrt(np.mean(signal**2))
    tcol = grid.time_grid[:, None]
    s_true = (sigma * (1.0 + 0.4 * np.cos(2.0 * np.pi * tcol))) ** 2  # (nt,1) broadcast
    s_true = np.broadcast_to(s_true, signal.shape)
    noise = rng.standard_normal(signal.shape) * np.sqrt(s_true)
    data = signal + noise
    print(f"[inject] coeffs {data.shape}  Mc={theta_true[0]:.0f}  "
          f"tc={theta_true[1]/86400:.2f}d  dL={params.distance:.0f}Mpc  "
          f"signal-rms/noise-rms={np.sqrt(np.mean(signal**2))/sigma:.2f}")

    # Initialise at the truth (GW chirp-phase posterior is extremely sharp in
    # Mc/tc; PE starts from a maximised point). The demo compares posterior
    # width/bias between noise models, not blind search.
    theta_ref = theta_true.copy()
    res = run_joint_sobh_wdm_mcmc(
        data, grid.time_grid, grid.freq_grid, signal_fn, theta_ref,
        config=cfg, theta_scale=np.array([0.02, 0.02 * params.tc, 0.5]),
        n_warmup=150, n_samples=150, random_seed=1,
    )

    for name, true in (("Mc", theta_true[0]), ("tc", theta_true[1]),
                       ("dL", params.distance)):
        s = res[name]
        z = (s.mean() - true) / s.std()
        print(f"[recover] {name}: post={s.mean():.4g} +/- {s.std():.3g}  "
              f"true={true:.4g}  z={z:+.2f}")
    surf_err = np.median(np.abs(np.log(res["psd_mean"]) - np.log(s_true)))
    print(f"[surface] median |log psd_mean - log S_true| = {surf_err:.3f}")
    print(f"[diag] divergences={res['divergences']}")


if __name__ == "__main__":
    main()
