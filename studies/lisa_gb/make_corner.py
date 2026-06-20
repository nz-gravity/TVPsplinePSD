"""Stationary-noise consistency corner: frequency-domain vs WDM GB posteriors.

Full nonlinear Galactic-binary parameter estimation (f0, fdot, A, phi0) on the A
and E TDI channels in stationary instrument noise, inferred in both the
frequency-domain Whittle likelihood and the WDM coefficient likelihood (reusing
the differentiable JaxGB PE of ``lisa_gb_study``). The two domains give
overlapping posteriors on the truth -- a consistency check that the
wavelet-domain source inference reproduces the standard frequency-domain answer,
so the joint signal+noise fit can be run in whichever domain the non-stationary
noise model is most convenient.

    uv run python studies/lisa_gb/make_corner.py            # reduced (~1-2 min)
    uv run python studies/lisa_gb/make_corner.py --production
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lisa_gb_study as S  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "notes" / "figures"
CACHE = Path(__file__).resolve().parents[2] / "studies" / "results" / "lisa" / "corner_samples.npz"
_LABELS = [r"$\log_{10} f_0$", r"$\dot f\,/\,10^{-15}$", r"$\log_{10} A$", r"$\phi_0$"]
_PARAMS = ("f0", "fdot", "A", "phi0")


def _matrix(p, phi_center):
    return np.column_stack([
        np.log10(p["f0"]), p["fdot"] / 1e-15, np.log10(p["A"]),
        phi_center + S.wrap_phase(p["phi0"] - phi_center),
    ])


def _render(posteriors: dict, truth: dict) -> None:
    """Overlay the freq-domain and WDM posteriors (dashed WDM so both show even
    when they coincide); truths in neutral grey."""
    import corner
    import matplotlib.pyplot as plt

    from tv_pspline_psd import set_paper_style
    set_paper_style()

    all_phi = np.concatenate([posteriors[d]["phi0"] for d in ("freq", "wdm")])
    pc = float(np.arctan2(np.mean(np.sin(all_phi)), np.mean(np.cos(all_phi))))
    truths = [np.log10(truth["f0"]), truth["fdot"] / 1e-15, np.log10(truth["A"]),
              float(pc + S.wrap_phase(truth["phi0"] - pc))]

    colors = {"freq": "tab:blue", "wdm": "tab:orange"}
    legend = {"freq": "Frequency domain", "wdm": "WDM"}
    styles = {"freq": "solid", "wdm": "dashed"}
    fig = None
    for dom in ("freq", "wdm"):
        fig = corner.corner(
            _matrix(posteriors[dom], pc), labels=_LABELS, truths=truths, fig=fig,
            color=colors[dom], truth_color="0.35", levels=(0.5, 0.9),
            plot_datapoints=False, plot_density=False, fill_contours=False,
            hist_kwargs={"density": True, "linestyle": styles[dom]},
            contour_kwargs={"linestyles": styles[dom]})
    fig.legend(handles=[plt.Line2D([0], [0], color=colors[d], ls=styles[d], label=legend[d])
                        for d in ("freq", "wdm")], loc="upper right", frameon=False, fontsize=12)
    out = FIG_DIR / "lisa_gb_stationary_corner.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[corner] saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--render-only", action="store_true",
                        help="Re-render the corner from cached posterior samples.")
    args = parser.parse_args()

    if args.render_only:
        d = np.load(CACHE)
        posteriors = {dom: {p: d[f"{dom}_{p}"] for p in _PARAMS} for dom in ("freq", "wdm")}
        _render(posteriors, {p: float(d[f"truth_{p}"]) for p in _PARAMS})
        return

    import jax
    jax.config.update("jax_enable_x64", True)
    S.CHANNELS = (0, 1)  # A and E only
    if not args.production:
        S.T_OBS = 120 * 24 * 3600
        S.NBLOCKS = 512
        S.N_WARMUP, S.N_DRAWS, S.NUM_CHAINS = 600, 800, 2

    grid = S._make_grid()
    jgb = S.make_jgb(grid)
    print(f"[corner] N={grid['n_total']} T_obs={grid['t_obs']/86400:.1f}d channels=A/E")

    # Per-seed source + SNR rescaling, exactly as lisa_gb_study.run_one_seed.
    rng = np.random.default_rng(args.seed)
    truth = S.draw_source(rng, grid)
    target_snr = float(rng.uniform(S.SNR_MIN, S.SNR_MAX))
    psd_full = S._psd_full(grid)
    scales = S._prior_scales(grid["t_obs"])
    prior_f0 = (truth["f0"] - scales["delta_f0_half"], truth["f0"] + scales["delta_f0_half"])
    sig_margin = S._signal_margin(grid, fdot=truth["fdot"], fdot_half=scales["delta_fdot_half"])
    bs = S._band_slices(grid, prior_f0[0] - sig_margin, prior_f0[1] + sig_margin)
    ref, _ = S.gb_full_rfft_np(jgb, grid, truth["f0"], truth["fdot"], 1.0, truth["phi0"], truth["sky"])
    truth["A"] = target_snr / np.sqrt(
        S._optimal_snr_sq(ref, psd_full, grid, band=slice(bs["kmin_rfft"], bs["kmax_rfft"])))
    print(f"[corner] seed {args.seed}: SNR={target_snr:.1f} f0={truth['f0']:.6e}")

    bands = S.build_band(grid, truth, args.seed, jgb)
    posteriors = {dom: S._samples(S.run_domain(bands[dom], args.seed + 10)) for dom in ("freq", "wdm")}
    for dom, p in posteriors.items():
        print(f"[corner] {dom}: A={np.median(p['A']):.3e} (truth {truth['A']:.3e})")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        **{f"{dom}_{p}": posteriors[dom][p] for dom in ("freq", "wdm") for p in _PARAMS},
        **{f"truth_{p}": truth[p] for p in _PARAMS})
    _render(posteriors, truth)


if __name__ == "__main__":
    main()
