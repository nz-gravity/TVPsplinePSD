"""Schematic contrasting WDM and Tang dynamic-Whittle observations.

Both observation models fit the *same* whitened P-spline surface; they differ only
in how they sample the time-frequency plane. This figure makes that contrast
visual:

  * Panel (a) -- the WDM transform tiles the plane into a regular, near-orthogonal
    grid, one chi^2_1 coefficient per cell.
  * Panel (b) -- a length-(2m+1) local DFT has m positive-frequency bins, but
    Tang retains only one bin at each successive window centre. The retained bin
    cycles from f_1 to f_m, making a diagonal block in a candidate pixel grid.
    Block starts are i*m centres apart, so (i-1)*m centres are skipped after each
    retained block.

Produces ``representation_schematic.png`` (and a vector ``.pdf``).

    python studies/paper_figures/scripts/make_representation_schematic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Palette and font sizes mirror the WDM-overview schematic for a consistent look.
COL = {
    "text":     "#1A1A1A",
    "text_dim": "#777777",
    "line":     "#1A1A1A",
    "grid":     "#C8C8C8",
    "wdm":      "#737373",   # shared monochrome accent for retained observations
    "observed": "#F1F1F1",   # parent WDM grid before configurable edge trimming
    "move":     "#737373",
    "even":     "#F4F4F4",
    "odd":      "#DCDCDC",
    "edge":     "#D0D0D0",
}
FS = {"panel": 9, "var": 10, "axis": 8, "sub": 7, "xlabel": 8}

# Small order and two-fold thinning keep the schematic readable.
TANG_M, TANG_THIN = 4, 2


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
                face = COL["observed"]
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

    _plane_axes(ax, x_lo=0.5, x_hi=nt - 0.5, y_lo=0.5, y_hi=nf + 0.5,
                x_axis_y=-0.62, y_axis_x=-0.95,
                xlab=r"normalised time $u=t/T$",
                ylab=r"normalised frequency $f/f_{\rm Nyq}$",
                x_lo_lab=r"$0$", x_hi_lab=r"$1$",
                y_lo_lab=r"$0$", y_hi_lab=r"$1$")

    panel_label(ax, r"$(a)$", r"WDM coefficients")


# ── Panel (b) — moving zigzag periodogram, scattered sampling ─────────────────
def draw_moving_zigzag(ax) -> None:
    clean_axis(ax)
    m, thin = TANG_M, TANG_THIN
    gap = (thin - 1) * m
    ncols = 2 * m + gap
    ax.set_xlim(-1.65, ncols + 0.82)
    ax.set_ylim(-1.85, m + 2.30)

    # Candidate local-Fourier grid. These pale cells are possible (t,f) pairs;
    # Tang never constructs the dense periodogram represented by all of them.
    for col in range(ncols):
        for row in range(m):
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#FAFAFA",
                                   edgecolor=COL["grid"], lw=0.35, zorder=0))

    # After m retained centres, (i-1)m intervening centres are not evaluated.
    ax.add_patch(Rectangle((m, 0), gap, m, facecolor="#E8E8E8",
                           edgecolor=COL["grid"], lw=0.5, zorder=1))
    ax.text(m + gap / 2, m / 2, "omitted", ha="center", va="center",
            fontsize=FS["sub"], color=COL["text_dim"], rotation=90, zorder=2)

    # One retained diagonal per block: at successive centres keep f_1,...,f_m.
    retained = [(j, j) for j in range(m)] + [(m + gap + j, j) for j in range(m)]
    for col, row in retained:
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=COL["move"],
                               edgecolor=COL["line"], lw=0.8, alpha=0.88, zorder=3))
    for start in (0, m + gap):
        ax.plot([start + 0.5, start + m - 0.5], [0.5, m - 0.5],
                color=COL["move"], lw=1.0, ls="--", alpha=0.65, zorder=4)

    # Row labels and block/gap annotations make the modulo rule explicit.
    for row in range(m):
        ax.text(-0.20, row + 0.5, rf"$f_{row + 1}$", ha="right", va="center",
                fontsize=FS["sub"], color=COL["text"])
    ax.text(m / 2, m + 0.18, r"cycle $f_1\rightarrow\cdots\rightarrow f_m$",
            ha="center", va="bottom", fontsize=FS["sub"], color=COL["text"])
    ax.text(m + gap + 0.12, m + 0.18, "reset",
            ha="left", va="bottom", fontsize=FS["sub"], color=COL["text_dim"])
    ax.text(ncols + 0.30, m / 2, r"$\cdots$", ha="center", va="center",
            fontsize=FS["axis"], color=COL["text_dim"])
    ax.text(-0.48, m + 0.33, r"$\vdots$", ha="center", va="center",
            fontsize=FS["axis"], color=COL["text_dim"])

    def bracket(x0, x1, y, label, color):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="|-|", lw=0.8, color=color))
        ax.text((x0 + x1) / 2, y - 0.20, label, ha="center", va="top",
                fontsize=FS["sub"], color=color)

    bracket(0.05, m - 0.05, -0.42, r"$m$ retained", COL["text"])
    bracket(m + 0.05, m + gap - 0.05, -0.42,
            r"$(i-1)m$ omitted", COL["text_dim"])

    # Shared physical axes: the displayed blocks are a schematic subset of the
    # normalised time-frequency domain.
    arr = dict(arrowstyle="->", lw=0.7, color=COL["text_dim"])
    ax.annotate("", xy=(ncols + 0.45, -1.18), xytext=(0.0, -1.18), arrowprops=arr)
    ax.text(0.0, -1.34, r"$0$", ha="center", va="top",
            fontsize=FS["sub"], color=COL["text"])
    ax.text(ncols + 0.45, -1.34, r"$1$", ha="center", va="top",
            fontsize=FS["sub"], color=COL["text"])
    ax.text((ncols + 0.45) / 2, -1.70, r"normalised time $u=t/T$",
            ha="center", va="center", fontsize=FS["axis"], color=COL["text"])
    ax.annotate("", xy=(-0.92, m + 0.62), xytext=(-0.92, 0.0), arrowprops=arr)
    ax.text(-0.78, 0.0, r"$0$", ha="left", va="center",
            fontsize=FS["sub"], color=COL["text"])
    ax.text(-0.78, m + 0.62, r"$1$", ha="left", va="center",
            fontsize=FS["sub"], color=COL["text"])
    ax.text(-1.45, (m + 0.62) / 2, r"normalised frequency $f/f_{\rm Nyq}$",
            rotation=90, ha="center", va="center",
            fontsize=FS["axis"], color=COL["text"])

    # Keep the transform definition compact; DFT symmetry and the precise
    # modulo index are explained in the caption rather than duplicated here.
    ax.text(ncols / 2, m + 1.27,
            r"$2m{+}1$ samples $\longrightarrow$ local DFT $\longrightarrow$ "
            r"$\{f_1,\ldots,f_m\}$; retain $f_{j(t)}$",
            ha="center", va="center", fontsize=FS["sub"], color=COL["text"])

    panel_label(ax, r"$(b)$", r"Thinned moving periodogram")


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
    # PRD two-column width; all typography is judged at this final size.
    fig = plt.figure(figsize=(7.1, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.96, 1.04],
                          left=0.045, right=0.99, bottom=0.04, top=0.97,
                          wspace=0.14)
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
