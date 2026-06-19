"""Stationary-row figure: frequency-domain vs WDM GB posteriors overlap.

Runs ONE seed of the stationary-noise galactic-binary study in both the
frequency-domain Whittle and the WDM-domain likelihoods, and overlays the two
4-parameter posteriors (f0, fdot, A, phi0) as a corner plot. The two domains
agreeing is the manuscript's stationary-noise consistency check: with a known
stationary PSD, the wavelet-domain analysis reproduces the standard
frequency-domain answer.

Reduced settings by default so it runs in ~1 minute; pass --production for the
one-year grid.

    uv run python studies/lisa_gb/make_corner.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lisa_gb_study as S  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--production", action="store_true",
                        help="Use the full one-year production grid/sampler settings.")
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    if not args.production:
        S.T_OBS = 30 * 24 * 3600
        S.NBLOCKS = 256
        S.N_WARMUP = 600
        S.N_DRAWS = 800
        S.NUM_CHAINS = 2

    grid = S._make_grid()
    jgb = S.make_jgb(grid)
    print(f"[corner] N={grid['n_total']} T_obs={grid['t_obs'] / 86400:.1f}d "
          f"dt={grid['dt']:.2f}s")

    # Draw the seed's source + SNR exactly as run_one_seed does, build the bands,
    # and sample both domains.
    rng = np.random.default_rng(args.seed)
    truth = S.draw_source(rng, grid)
    target_snr = float(rng.uniform(S.SNR_MIN, S.SNR_MAX))
    psd_full = S._psd_full(grid)
    scales = S._prior_scales(grid["t_obs"])
    prior_f0 = (truth["f0"] - scales["delta_f0_half"], truth["f0"] + scales["delta_f0_half"])
    sig_margin = S._signal_margin(grid, fdot=truth["fdot"], fdot_half=scales["delta_fdot_half"])
    bs = S._band_slices(grid, prior_f0[0] - sig_margin, prior_f0[1] + sig_margin)
    ref_rfft, _ = S.gb_full_rfft_np(jgb, grid, truth["f0"], truth["fdot"], 1.0,
                                    truth["phi0"], truth["sky"])
    snr0 = np.sqrt(S._optimal_snr_sq(ref_rfft, psd_full, grid,
                                     band=slice(bs["kmin_rfft"], bs["kmax_rfft"])))
    truth["A"] = target_snr / snr0
    print(f"[corner] seed {args.seed}: SNR={target_snr:.1f} f0={truth['f0']:.6e}")

    bands = S.build_band(grid, truth, args.seed, jgb)
    posteriors = {}
    for domain in ("freq", "wdm"):
        mcmc = S.run_domain(bands[domain], args.seed + 10)
        posteriors[domain] = S._samples(mcmc)  # dict f0,fdot,A,phi0 -> (n,)
        print(f"[corner] {domain} done (div="
              f"{int(np.asarray(mcmc.get_extra_fields()['diverging']).sum())})")

    # Stack into the displayed marginals (log10 f0, fdot/1e-15, log10 A, phi0).
    def _matrix(p):
        return np.column_stack([
            np.log10(p["f0"]),
            p["fdot"] / 1e-15,
            np.log10(p["A"]),
            S.wrap_phase(p["phi0"]),
        ])

    labels = [r"$\log_{10} f_0$", r"$\dot f\,/\,10^{-15}$", r"$\log_{10} A$", r"$\phi_0$"]
    truths = [np.log10(truth["f0"]), truth["fdot"] / 1e-15,
              np.log10(truth["A"]), float(S.wrap_phase(truth["phi0"]))]

    import corner
    import matplotlib.pyplot as plt

    fig = corner.corner(_matrix(posteriors["freq"]), labels=labels, truths=truths,
                        color="tab:blue", hist_kwargs={"density": True},
                        plot_datapoints=False, plot_density=False, fill_contours=False)
    corner.corner(_matrix(posteriors["wdm"]), fig=fig, color="tab:orange",
                  hist_kwargs={"density": True}, plot_datapoints=False,
                  plot_density=False, fill_contours=False)
    fig.legend(handles=[
        plt.Line2D([0], [0], color="tab:blue", label="Frequency domain"),
        plt.Line2D([0], [0], color="tab:orange", label="WDM"),
    ], loc="upper right", frameon=False, fontsize=12)
    out = Path(__file__).resolve().parents[1].parent / "notes" / "figures" / "lisa_gb_stationary_corner.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[corner] saved {out}")


if __name__ == "__main__":
    main()
