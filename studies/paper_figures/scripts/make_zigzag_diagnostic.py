"""Diagnostic figure: what the zigzag is, and what thinning actually does.

Written to answer two review points on the moving-periodogram section:

  (a) "The zigzag does not come from thinning but from the definition of
      f_mod(t)."  Panel (a) plots the evaluated frequency index against the
      window centre for i = 1, 2, 3.  The sawtooth is fully present at i = 1
      (no thinning at all); larger i keeps the *same* sawtooth and deletes
      whole blocks from it.

  (b) The two boundary deviations from Tang Definition 3, recorded in the
      module docstring of ``tv_pspline_psd/moving_periodogram.py``.  Panel (b)
      shows, per i, which samples are covered by a retained window and which
      are never used at all.

Every point plotted is read back from ``tang_moving_periodogram`` itself -- no
value here is hand-placed -- so the figure shows what the implementation does
rather than what it is meant to do.  Note the limit of that: a picture can show
*which* (t, f) pairs are evaluated, but not whether eq. (2.2) is normalised or
phase-signed correctly.  Those are pinned by the paper-faithful oracle in
``tests/test_moving_periodogram.py`` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from tv_pspline_psd.moving_periodogram import tang_moving_periodogram

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Short series and small order so the whole record fits one shared time axis:
# every panel then lines up column-for-column, and the m-sample head offset is
# wide enough on the page to actually see.
T, M = 200, 12
THINS = (1, 2, 3)

COL = {
    "text": "#1A1A1A",
    "dim": "#8A8A8A",
    "kept": "#2A5D8F",
    "dropped": "#C8C8C8",
    "unused": "#B4433A",
    "used": "#9FB8CE",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })


def ordinate_times(x: np.ndarray, thin: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centre index t, 1-based frequency index j) straight from the code."""
    out = tang_moving_periodogram(x, m=M, thin=thin)
    t = np.rint(out["u"] * x.size).astype(int)
    # Recover j from omega = pi * 2j/(2m+1); exact for the rungs actually used.
    j = np.rint(out["omega"] * (2 * M + 1) / (2.0 * np.pi)).astype(int)
    return t, j


def draw_zigzag(axes, x: np.ndarray) -> None:
    """Panel (a): the sawtooth exists at i=1; thinning deletes blocks from it."""
    t_ref, j_ref = ordinate_times(x, 1)

    for ax, thin in zip(axes, THINS):
        # The unthinned sawtooth, as a faint backdrop identical in every row.
        ax.plot(t_ref, j_ref, marker="o", ms=1.8, lw=0.5,
                color=COL["dropped"], zorder=1)

        t, j = ordinate_times(x, thin)
        ax.plot(t, j, ls="none", marker="o", ms=3.0,
                color=COL["kept"], zorder=3)

        ax.set_ylabel(rf"$i={thin}$", fontsize=8.5, rotation=0,
                      ha="right", va="center", labelpad=8)
        ax.set_ylim(-0.6, M + 1.6)
        ax.set_yticks([1, M])
        ax.set_yticklabels([r"$f_1$", rf"$f_{{{M}}}$"], fontsize=7)
        ax.tick_params(labelbottom=False, length=2.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_title(
        r"$(a)$  the zigzag is $\mathrm{mod}(t)=1+((t-1)\ \mathrm{mod}\ m)$"
        "—" r"present already at $i=1$, so it is not made by thinning",
        fontsize=8.5, loc="left", style="italic", pad=14)
    axes[0].legend(
        handles=[
            plt.Line2D([], [], marker="o", ms=3.0, ls="none",
                       color=COL["kept"], label=r"evaluated at this $i$"),
            plt.Line2D([], [], marker="o", ms=1.8, lw=0.5,
                       color=COL["dropped"],
                       label=r"unthinned ($i=1$) sawtooth, for reference"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=6.8, ncol=2,
        frameon=False, borderaxespad=0.0, handletextpad=0.4, columnspacing=1.4)


def draw_coverage(ax, x: np.ndarray) -> None:
    """Panel (b): which samples reach the likelihood, and which never do."""
    for row, thin in enumerate(THINS):
        t, _ = ordinate_times(x, thin)
        first_touched = t.min() - M       # earliest sample inside any window
        last_touched = t.max() + M        # latest sample inside any window
        y = len(THINS) - 1 - row

        # Samples that enter at least one retained window.
        ax.add_patch(Rectangle((first_touched, y - 0.3),
                               last_touched - first_touched, 0.6,
                               facecolor=COL["used"], edgecolor="none", zorder=2))
        # Range of window *centres* -- starts at m+1, never at Tang's t=1.
        ax.plot([t.min(), t.max()], [y, y], lw=1.6, color=COL["kept"],
                solid_capstyle="butt", zorder=3)
        # Trailing samples that no retained window ever covers.
        if last_touched < T:
            ax.add_patch(Rectangle((last_touched, y - 0.3), T - last_touched, 0.6,
                                   facecolor=COL["unused"], edgecolor="none",
                                   zorder=2))
            ax.text(T + 6, y, rf"$-{T - last_touched}$", va="center", ha="left",
                    fontsize=7, color=COL["unused"])

        ax.text(-8, y, rf"$i={thin}$", va="center", ha="right", fontsize=8)

    # Deviation 1: no centre before m+1, because we never pad.
    ax.axvspan(0.5, M + 1, facecolor=COL["dim"], alpha=0.16, zorder=0)
    ax.annotate(r"no centre here: $t<m{+}1$ would need padding "
                r"$X_{-m+1},\ldots$ (Tang Def. 3)",
                xy=(M + 1, len(THINS) - 0.30), xytext=(M + 20, len(THINS) + 0.42),
                fontsize=6.8, color=COL["text"], va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=COL["dim"]))

    ax.set_xlim(-2, T + 26)
    ax.set_ylim(-0.75, len(THINS) + 0.85)
    ax.set_yticks([])
    ax.set_xlabel(r"sample index  ($T=%d$, $m=%d$)" % (T, M), labelpad=2)
    ax.set_title(
        r"$(b)$  boundary deviations: centres start at $m{+}1$, "
        r"and a tail of $((T{-}2m)\ \mathrm{mod}\ im)+m(i{-}1)$ samples is never used",
        fontsize=8.5, loc="left", style="italic", pad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5)
    ax.legend(handles=[
        Patch(facecolor=COL["used"], label="samples inside a retained window"),
        plt.Line2D([], [], color=COL["kept"], lw=1.6, label=r"range of centres $t$"),
        Patch(facecolor=COL["unused"], label="never used (dropped tail)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.46), fontsize=6.8, ncol=3,
        frameon=False, borderaxespad=0.0, handletextpad=0.5, columnspacing=1.4)


def draw_grid_contrast(ax_wdm, ax_mp) -> None:
    """Why log S cannot be assembled as B_t W B_f^T for the moving periodogram.

    Both panels hold the same number of observations over the same box.  WDM
    spends them as *few times x every frequency* -- a filled rectangle, which is
    exactly what the two dense matrix products produce.  Tang Definition 1 gives
    one ordinate per time point, so the same budget is spent as *many times x one
    frequency each*: a thread through the plane, not a rectangle.
    """
    n_f, n_t_wdm = 6, 4
    n_obs = n_f * n_t_wdm                      # identical budget in both panels

    # WDM: every (time, frequency) cell of a coarse time grid is observed.
    for it in range(n_t_wdm):
        for jf in range(n_f):
            ax_wdm.add_patch(Rectangle(
                (it * n_f, jf), n_f, 1, facecolor=COL["used"],
                edgecolor="white", lw=1.0, zorder=2))
    ax_wdm.set_title(
        rf"WDM: {n_t_wdm} times $\times$ {n_f} frequencies $=$ {n_obs}"
        "\n" r"every frequency at every time $\Rightarrow$ "
        r"$\log S=\mathbf{B}_t\mathbf{W}\mathbf{B}_f^\top$",
        fontsize=7.6, loc="left", pad=5)

    # Moving periodogram: one ordinate per time, at frequency mod(t).
    t = np.arange(n_obs)
    j = t % n_f
    ax_mp.plot(t + 0.5, j + 0.5, marker="o", ms=3.4, lw=0.6,
               color=COL["kept"], zorder=3)
    ax_mp.set_title(
        rf"moving periodogram: {n_obs} times $\times$ 1 frequency $=$ {n_obs}"
        "\n" r"one frequency per time $\Rightarrow$ a thread, not a rectangle",
        fontsize=7.6, loc="left", pad=5)

    for ax in (ax_wdm, ax_mp):
        for jf in range(n_f + 1):
            ax.axhline(jf, color=COL["dropped"], lw=0.4, zorder=1)
        ax.set_xlim(0, n_obs)
        ax.set_ylim(0, n_f)
        ax.set_xticks([])
        ax.set_yticks(np.arange(n_f) + 0.5)
        ax.set_yticklabels([rf"$f_{{{k + 1}}}$" for k in range(n_f)], fontsize=6.5)
        ax.set_xlabel("time", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)


def save_grid_contrast(outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.95))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.74, wspace=0.16)
    draw_grid_contrast(*axes)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"grid_vs_thread.{ext}",
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved to {outdir / 'grid_vs_thread.png'}")


def save_figure(outdir: Path) -> None:
    x = np.random.default_rng(0).standard_normal(T)
    fig = plt.figure(figsize=(7.1, 5.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.30],
                          left=0.085, right=0.985, bottom=0.165, top=0.895,
                          hspace=0.42)
    # One shared time axis, so the sawtooth in (a) lines up column-for-column
    # with the coverage bars in (b).
    top = fig.add_subplot(gs[0, 0])
    zig = [top] + [fig.add_subplot(gs[i, 0], sharex=top) for i in (1, 2)]
    cov = fig.add_subplot(gs[3, 0], sharex=top)
    draw_zigzag(zig, x)
    draw_coverage(cov, x)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"zigzag_diagnostic.{ext}",
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved to {outdir / 'zigzag_diagnostic.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    set_style()
    save_figure(args.outdir)
    save_grid_contrast(args.outdir)
