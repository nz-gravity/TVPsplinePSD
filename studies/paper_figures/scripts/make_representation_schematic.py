"""Schematic contrasting the two time-frequency front ends (manuscript Section 3).

Both observation models fit the *same* whitened P-spline surface; they differ only
in how they sample the time-frequency plane. This figure makes that contrast
visual:

  * Panel (a) -- the WDM transform tiles the plane into a regular, near-orthogonal
    grid, one chi^2_1 coefficient per cell.
  * Panel (b) -- the Tang zigzag moving periodogram does not fill the plane: at
    each step the evaluated Fourier frequency cycles with the time index, tracing
    rising diagonal ramps, and the thinning skips i*m ordinates between blocks,
    leaving a scattered sawtooth of chi^2_2 ordinates each carrying a 2m+1 window.

The scattered (u, omega) ordinates in panel (b) are produced by the genuine
``tang_moving_periodogram`` construction, so the sampling geometry is faithful.

Produces ``representation_schematic.png`` (and a vector ``.pdf``).

    python studies/paper_figures/scripts/make_representation_schematic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from tv_pspline_psd import tang_moving_periodogram

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Palette and font sizes mirror the WDM-overview schematic for a consistent look.
COL = {
    "text":     "#1A1A1A",
    "text_dim": "#777777",
    "line":     "#1A1A1A",
    "grid":     "#C8C8C8",
    "wdm":      "#4C7A9A",   # muted slate blue -- WDM
    "move":     "#A65F3A",   # muted rust       -- moving periodogram
    "even":     "#F4F4F4",
    "odd":      "#DCDCDC",
    "edge":     "#BEBEBE",
}
FS = {"panel": 11, "var": 13, "axis": 10, "sub": 8.5, "xlabel": 10}

# Faithful moving-periodogram sampling: small order, two-fold thinning. A short
# white series gives a clean, readable handful of zigzag ramps.
TANG_M, TANG_THIN = 6, 2
_SERIES_LEN = 6 * TANG_THIN * TANG_M + 2 * TANG_M  # exactly six blocks


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": FS["axis"],
        "axes.labelsize": FS["xlabel"],
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_label(ax, letter: str, title: str) -> None:
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS["panel"], fontstyle="italic", color=COL["text"])
    ax.text(0.105, 0.995, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS["var"], color=COL["text"])


# ── Panel (a) — WDM regular tiling ────────────────────────────────────────────
def draw_wdm_grid(ax, *, nt: int = 10, nf: int = 6) -> None:
    clean_axis(ax)
    ax.set_xlim(-1.7, nt + 0.4)
    ax.set_ylim(-1.5, nf + 2.5)

    for n in range(nt):
        for m in range(nf + 1):
            if m in (0, nf):
                face = COL["edge"]
            else:
                face = COL["even"] if (n + m) % 2 == 0 else COL["odd"]
            ax.add_patch(Rectangle((n, m), 1, 1, facecolor=face,
                                   edgecolor=COL["grid"], lw=0.4))
    ax.add_patch(Rectangle((0, 0), nt, nf + 1, facecolor="none",
                           edgecolor=COL["line"], lw=1.0))

    # One representative cell -- a single chi^2_1 coefficient.
    n0, m0 = 6, 4
    ax.add_patch(Rectangle((n0, m0), 1, 1, facecolor=COL["wdm"],
                           edgecolor=COL["line"], lw=1.0, alpha=0.85, zorder=5))
    ax.annotate(r"$w_{nm}\sim\mathcal{N}(0,\,S_{nm})$",
                xy=(n0 + 0.5, m0 + 1.0), xytext=(n0 - 1.4, m0 + 2.15),
                ha="center", va="bottom", fontsize=FS["sub"], color=COL["text"],
                arrowprops=dict(arrowstyle="-", lw=0.8, color=COL["line"]))

    # Tile dimensions.
    ax.annotate("", xy=(2, 1.5 - 0.30), xytext=(3, 1.5 - 0.30),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=COL["text_dim"]))
    ax.text(2.5, 1.5 - 0.46, r"$\Delta T$", ha="center", va="top",
            fontsize=FS["sub"], color=COL["text_dim"])
    ax.annotate("", xy=(2 - 0.30, 1), xytext=(2 - 0.30, 2),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=COL["text_dim"]))
    ax.text(2 - 0.44, 1.5, r"$\Delta F$", ha="right", va="center",
            fontsize=FS["sub"], color=COL["text_dim"])

    _plane_axes(ax, x_lo=0.5, x_hi=nt - 0.5, y_lo=0.5, y_hi=nf + 0.5,
                x_axis_y=-0.62, y_axis_x=-0.95,
                xlab=r"time bins $n$", ylab=r"frequency channels $m$",
                x_lo_lab=r"$0$", x_hi_lab=r"$N_t{-}1$",
                y_lo_lab=r"$0$", y_hi_lab=r"$N_f$")

    for m_edge, lbl in [(0.5, "DC"), (nf + 0.5, "Nyquist")]:
        ax.text(nt - 0.12, m_edge, lbl, ha="right", va="center",
                fontsize=FS["sub"], color=COL["text_dim"])

    panel_label(ax, r"$(a)$", r"WDM: regular tiling")


# ── Panel (b) — moving zigzag periodogram, scattered sampling ─────────────────
def draw_moving_zigzag(ax) -> None:
    clean_axis(ax)
    rng = np.random.default_rng(0)
    ordinates = tang_moving_periodogram(
        rng.standard_normal(_SERIES_LEN), m=TANG_M, thin=TANG_THIN)
    u = ordinates["u"]
    rung = ordinates["omega"] / np.pi  # m Fourier rungs in (0, 1)

    ax.set_xlim(-0.10, 1.06)
    ax.set_ylim(-0.235, 1.34)

    # Faint frequency rungs: ordinates only ever land on these m rows.
    for r in np.unique(rung):
        ax.plot([0.0, 1.0], [r, r], color=COL["grid"], lw=0.5, ls=(0, (1, 2)),
                zorder=0)

    # One 2m+1 window, shaded behind the first ramp's lowest ordinate.
    t_demo = TANG_M + TANG_THIN * TANG_M + 1     # block 1, first ordinate
    win_lo, win_hi = (t_demo - TANG_M) / _SERIES_LEN, (t_demo + TANG_M) / _SERIES_LEN
    win_mid = 0.5 * (win_lo + win_hi)
    ax.axvspan(win_lo, win_hi, color=COL["move"], alpha=0.10, lw=0, zorder=1)
    ax.annotate(r"$2m{+}1$ window", xy=(win_mid, 1.01),
                xytext=(win_mid, 1.10), ha="center", va="bottom",
                fontsize=FS["sub"], color=COL["move"],
                arrowprops=dict(arrowstyle="-", lw=0.7, color=COL["move"]))

    # Rising ramps (one block = one ramp), then a thinning gap and reset.
    block = TANG_M
    for b in range(u.size // block):
        sl = slice(b * block, (b + 1) * block)
        ax.plot(u[sl], rung[sl], color=COL["move"], lw=1.1, ls="--", alpha=0.55,
                zorder=2)
    ax.scatter(u, rung, s=22, facecolor=COL["move"], edgecolor=COL["line"],
               linewidth=0.5, zorder=3)

    ax.annotate(r"$\mathrm{MI}_t\sim\mathrm{Exp}(1/S)$",
                xy=(u[block + block - 1], rung[block + block - 1]),
                xytext=(0.47, 1.13), ha="left", va="bottom",
                fontsize=FS["sub"], color=COL["text"],
                arrowprops=dict(arrowstyle="-", lw=0.8, color=COL["line"]))
    ax.annotate(r"thinning gap ($i\,m$ skipped)",
                xy=(0.5 * (u[block - 1] + u[block]), rung[0] - 0.045),
                xytext=(0.62, -0.045), ha="left", va="center",
                fontsize=FS["sub"], color=COL["text_dim"],
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["text_dim"]))

    _plane_axes(ax, x_lo=0.0, x_hi=1.0, y_lo=0.0, y_hi=1.0,
                x_axis_y=-0.12, y_axis_x=-0.055,
                xlab=r"time $t$", ylab=r"frequency $f$",
                x_lo_lab=r"$0$", x_hi_lab=r"$T$",
                y_lo_lab=r"$0$", y_hi_lab=r"$f_{\max}$")

    panel_label(ax, r"$(b)$", r"moving zigzag periodogram")


# ── shared light-weight plane axes (arrows + four corner labels) ──────────────
def _plane_axes(ax, *, x_lo, x_hi, y_lo, y_hi, x_axis_y, y_axis_x,
                xlab, ylab, x_lo_lab, x_hi_lab, y_lo_lab, y_hi_lab) -> None:
    arr = dict(arrowstyle="->", lw=0.7, color=COL["text_dim"])
    ax.annotate("", xy=(x_hi, x_axis_y), xytext=(x_lo, x_axis_y), arrowprops=arr)
    ax.annotate("", xy=(y_axis_x, y_hi), xytext=(y_axis_x, y_lo), arrowprops=arr)
    span_x, span_y = x_hi - x_lo, y_hi - y_lo
    ax.text(x_lo, x_axis_y - 0.045 * span_y - 0.04, x_lo_lab,
            ha="center", va="top", fontsize=FS["sub"], color=COL["text"])
    ax.text(x_hi, x_axis_y - 0.045 * span_y - 0.04, x_hi_lab,
            ha="center", va="top", fontsize=FS["sub"], color=COL["text"])
    ax.text(0.5 * (x_lo + x_hi), x_axis_y - 0.12 * span_y - 0.04, xlab,
            ha="center", va="top", fontsize=FS["axis"], color=COL["text"])
    ax.text(y_axis_x - 0.02 * span_x, y_lo, y_lo_lab, ha="right", va="center",
            fontsize=FS["sub"], color=COL["text"])
    ax.text(y_axis_x - 0.02 * span_x, y_hi, y_hi_lab, ha="right", va="center",
            fontsize=FS["sub"], color=COL["text"])
    ax.text(y_axis_x - 0.085 * span_x, 0.5 * (y_lo + y_hi), ylab,
            ha="center", va="center", rotation="vertical",
            fontsize=FS["axis"], color=COL["text"])


def save_figure(outdir: Path) -> None:
    fig = plt.figure(figsize=(8.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          left=0.045, right=0.985, bottom=0.04, top=0.97,
                          wspace=0.16)
    draw_wdm_grid(fig.add_subplot(gs[0, 0]))
    draw_moving_zigzag(fig.add_subplot(gs[0, 1]))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"representation_schematic.{ext}",
                    bbox_inches="tight", pad_inches=0.05, transparent=False)
    plt.close(fig)
    print(f"Saved to {outdir / 'representation_schematic.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    set_style()
    save_figure(args.outdir)
