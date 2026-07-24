#!/usr/bin/env python3
"""Physics map of the constrained field and its symplectic hierarchy.

Panel (a) separates the determinant-null contour motion from the fractional
response required by the constraint.  Panel (b) shows the two distinct
sources at Lambda=7/4 and Lambda=11/4 that enter the audited restricted
interaction at the Lambda=13/4 opening resonance.  It is a conceptual map,
not a plot of numerical data.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
FIG = HERE / "rendered"
INK = "#1a1a1a"
MUTED = "#707070"
GRID = "#dedede"
MR = "#c1272d"
BLUE = "#0072B2"

plt.rcParams.update({
    "font.size": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def panel_label(ax, letter: str, title: str) -> None:
    ax.text(
        0.005, 1.025, letter, transform=ax.transAxes,
        fontsize=12.2, fontweight="bold", va="bottom",
    )
    ax.text(
        0.050, 1.025, title, transform=ax.transAxes,
        fontsize=11.6, va="bottom",
    )


def node(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    formula: str,
    note: str,
    *,
    edge: str = INK,
    fill: str = "white",
    linestyle: str = "-",
    title_color: str | None = None,
    title_size: float = 10.1,
    formula_size: float = 10.0,
    note_size: float = 8.7,
) -> None:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        facecolor=fill, edgecolor=edge, linewidth=1.45,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    cx = x + width / 2
    ax.text(
        cx, y + 0.79 * height, title,
        ha="center", va="center", fontsize=title_size, fontweight="bold",
        color=title_color or edge,
    )
    ax.text(
        cx, y + 0.51 * height, formula,
        ha="center", va="center", fontsize=formula_size,
        color=INK, linespacing=1.05,
    )
    ax.text(
        cx, y + 0.19 * height, note,
        ha="center", va="center", fontsize=note_size,
        color=edge, linespacing=1.08,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9,
        linewidth=1.05, color=color, connectionstyle=connectionstyle,
        shrinkA=1.5, shrinkB=1.5,
    ))
    if label and label_xy:
        ax.text(
            *label_xy, label, ha="center", va="center",
            fontsize=8.1, color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )


def make_figure() -> None:
    fig = plt.figure(figsize=(7.25, 6.05))
    gs = fig.add_gridspec(
        2, 1, height_ratios=(0.92, 1.45),
        left=0.025, right=0.988, top=0.95, bottom=0.050, hspace=0.34,
    )

    # (a) The constraint fixes a leading orbit, but it leaves an independent
    # contour motion.  Do not draw Cs as the cause of the 5/4 residual.
    axa = fig.add_subplot(gs[0])
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.axis("off")
    panel_label(axa, "a", "constraint-active near-tip field")

    node(
        axa, 0.015, 0.27, 0.245, 0.52,
        "constitutive stiffening",
        r"$c_2I_2\simeq c_2\lambda_1^2$"
        "\n" r"$(\lambda_2^2+\lambda_3^2)$",
        "acts as a transverse penalty",
        fill="#fafafa", formula_size=9.3,
    )
    node(
        axa, 0.355, 0.22, 0.285, 0.62,
        "constraint + reaction",
        r"$\lambda_2=\lambda_3=\lambda_1^{-1/2}$"
        "\n" r"$J^4=|\nabla y_2|^2$"
        "\n" r"$y_2\sim P r^{1/2}\sin(\theta/2)$",
        "leading uniaxial orbit;\ncarries the energy flux",
        edge=INK, fill="#fafafa", formula_size=9.2, note_size=8.2,
    )
    node(
        axa, 0.735, 0.57, 0.25, 0.31,
        "allowed contour motion",
        r"$F(y_2)\supset C_s s\sim r$",
        "outer-selected"
        "\n" r"leaves $J$ and leading $G$ unchanged",
        edge=BLUE, fill="#f5faff", formula_size=9.8, note_size=7.9,
    )
    node(
        axa, 0.735, 0.06, 0.25, 0.38,
        r"$r^{5/4}$ fractional response",
        r"$y_1-c_0-F(y_2)$"
        "\n" r"$\sim P^{-1/2}r^{5/4}$"
        r"$[g_a+C_h f^{5/2}]$",
        r"axis FEM: $q=1.2515$"
        "\n" r"$A_{\rm ax}/A_{\rm ax,pred}=1.0124$",
        edge=MR, fill="#fff7f7", title_size=9.0,
        formula_size=8.2, note_size=7.8,
    )
    arrow(axa, (0.260, 0.53), (0.355, 0.53))
    arrow(
        axa, (0.640, 0.60), (0.735, 0.715),
        color=BLUE, label="null freedom", label_xy=(0.686, 0.765),
        connectionstyle="arc3,rad=-0.10",
    )
    arrow(
        axa, (0.640, 0.43), (0.735, 0.26),
        color=MR, connectionstyle="arc3,rad=0.10",
    )

    # (b) The 7/4 material correction and the 11/4 stationary-background
    # checkpoint are distinct sources.  Their restricted nonlinear
    # interaction meets the Lambda=13/4 opening kernel.
    axb = fig.add_subplot(gs[1])
    axb.set_xlim(0, 1)
    axb.set_ylim(-0.16, 1)
    axb.axis("off")
    panel_label(
        axb, "b",
        "symplectic hierarchy: corrections, matching, and resonance",
    )

    node(
        axb, 0.015, 0.35, 0.19, 0.36,
        r"selected $F=0$ base",
        "analytic-axis\nrepresentative",
        "bookkeeping choice,"
        "\nnot specimen selection",
        edge=INK, fill="#fafafa", title_size=9.4,
        formula_size=9.2, note_size=7.8,
    )
    node(
        axb, 0.295, 0.59, 0.285, 0.32,
        r"closed $\Lambda=7/4$",
        r"first $c_2/c_1$ material correction"
        "\n" r"$\delta y_2\sim r,\quad\delta y_1\sim r^{7/4}$",
        "total traction closes;"
        "\nno new local amplitude",
        edge=MR, fill="#fff4f4", formula_size=8.8, note_size=7.8,
    )
    node(
        axb, 0.295, 0.10, 0.285, 0.32,
        r"formal $\Lambda=11/4$",
        "stationary-background response"
        "\n" r"opening power $r^2$",
        "outer endpoint checkpoint;"
        "\nphysical matching open",
        edge=MUTED, fill="white", linestyle=(0, (4, 2)),
        formula_size=8.9, note_size=7.8,
    )

    # Merge point for the audited restricted interaction.
    merge_x, merge_y = 0.655, 0.51
    axb.plot(
        merge_x, merge_y, "o", ms=7.5, mfc="white", mec=INK, mew=1.4,
        zorder=5,
    )
    axb.text(
        merge_x, 0.37,
        "restricted\ninteraction",
        ha="center", va="top", fontsize=7.8, color=INK, linespacing=1.05,
    )

    node(
        axb, 0.735, 0.29, 0.25, 0.48,
        "resonant opening mode",
        r"$\Lambda=13/4$: kernel $\sin(5\theta/2)$"
        "\n" r"$(\delta y_2)_{\rm restricted}$"
        "\n" r"$\propto r^{5/2}\log(r/r_0)$"
        "\n" r"$\times\sin(5\theta/2)$",
        "nonzero projection forces this log;"
        "\ncomplete source and coupled"
        "\nresponse remain open",
        edge=MR, fill="white", title_size=9.7,
        formula_size=7.9, note_size=7.2,
    )

    arrow(
        axb, (0.205, 0.57), (0.295, 0.75),
        color=MR,
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        axb, (0.205, 0.48), (0.295, 0.26),
        color=MUTED,
        connectionstyle="arc3,rad=0.12",
    )
    arrow(axb, (0.580, 0.75), (merge_x, merge_y), color=MR)
    arrow(axb, (0.580, 0.26), (merge_x, merge_y), color=MUTED)
    arrow(axb, (0.670, merge_y), (0.735, merge_y), color=MR)

    # The exact local seeds belong to parallel matching sectors, not to the
    # forced material/background chain above.
    band = FancyBboxPatch(
        (0.145, -0.125), 0.84, 0.13,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#f7f7f7", edgecolor=GRID, linewidth=1.0,
    )
    axb.add_patch(band)
    axb.text(
        0.565, -0.058,
        r"Parallel matching sectors: opening harmonic "
        r"$B\,r^{3/2}\sin(3\theta/2)$ and contour shears $Q_k s^k$"
        "\n" r"are exact local seeds; the outer specimen supplies their amplitudes.",
        ha="center", va="center", fontsize=8.4, color=MUTED,
        linespacing=1.05,
    )

    FIG.mkdir(exist_ok=True)
    fig.savefig(
        FIG / "fig_asymap.pdf",
        metadata={
            "Creator": "make_fig_asymap.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(FIG / "fig_asymap.png")
    plt.close(fig)
    print(f"wrote {FIG}/fig_asymap.{{pdf,png}}")


if __name__ == "__main__":
    make_figure()
