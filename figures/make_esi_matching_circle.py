#!/usr/bin/env python3
"""Render the ESI matching-circle field and radial-window map.

The contour is read directly from the final retained P2 polar profile.  No
derivatives are reconstructed and no finite-element solve is run here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def repository_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "fem").is_dir():
            return candidate
    raise FileNotFoundError("could not locate repository root")


ROOT = repository_root()
PROFILE_CANDIDATES = (
    ROOT / "fem" / "audit_data" / "global_local"
    / "global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz",
    ROOT / "data" / "fem" / "global_local"
    / "global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz",
)
PROFILE = next(
    (candidate for candidate in PROFILE_CANDIDATES if candidate.is_file()),
    None,
)
if PROFILE is None:
    raise FileNotFoundError("final matching-circle profile is missing")

OUTPUT = HERE / ("rendered" if HERE.name == "figures" else "figures")

WINDOWS = (
    ("I", 1.5e-4, 1.6e-3, "#666666"),
    ("II", 3.0e-4, 3.0e-3, "#0072B2"),
    ("III", 6.0e-4, 3.8e-3, "#c1272d"),
)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.02, 0.98, letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=12.5, fontweight="bold",
    )


def make_figure() -> None:
    profile = np.load(PROFILE)
    X = np.asarray(profile["X"], dtype=float)
    Y = np.asarray(profile["Y"], dtype=float)
    Y2 = np.asarray(profile["Y2"], dtype=float)
    r_min = float(profile["r_min"])
    matching_radius = float(profile["matching_radius"])
    sample_lo = float(np.min(profile["r"]))
    sample_hi = float(np.max(profile["r"]))

    plt.rcParams.update({
        "font.size": 10.0,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    fig, (ax_field, ax_windows) = plt.subplots(
        2, 1, figsize=(7.2, 5.85),
        gridspec_kw={"height_ratios": (1.38, 1.0), "hspace": 0.46},
    )

    # (a) Direct contour of the retained finite-element field in reference
    # coordinates.  The stored profile stops at 0.8 R_m.
    levels = np.linspace(float(np.nanmin(Y2)), float(np.nanmax(Y2)), 19)
    contour = ax_field.contourf(
        X, Y, Y2, levels=levels, cmap="cividis", extend="both",
    )
    theta = np.linspace(0.0, np.pi, 500)
    ax_field.plot(
        matching_radius * np.cos(theta),
        matching_radius * np.sin(theta),
        color="0.12", lw=1.6, ls=(0, (5, 2.5)),
        label=r"matching interface $R_m$",
    )
    ax_field.plot(
        sample_hi * np.cos(theta), sample_hi * np.sin(theta),
        color="white", lw=0.9, ls=":", alpha=0.95,
    )
    ax_field.plot(
        [sample_lo, matching_radius], [0.0, 0.0],
        color="#c1272d", lw=2.0,
        label=r"fitted axis $\theta=0$",
    )
    ax_field.plot(
        [-matching_radius, -sample_lo], [0.0, 0.0],
        color="0.15", lw=1.3,
    )
    ax_field.annotate(
        r"complete P2 trace at $R_m/h=0.01$",
        xy=(matching_radius * np.cos(np.pi / 3),
            matching_radius * np.sin(np.pi / 3)),
        xytext=(-0.0096, 0.0101), fontsize=8.7,
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "0.2"},
    )
    ax_field.text(
        0.0020, 0.00055, r"exact-axis samples", color="#a51f25",
        fontsize=8.5, ha="left", va="bottom",
    )
    ax_field.text(
        -0.0094, 0.00055, "crack face", color="0.15",
        fontsize=8.5, ha="left", va="bottom",
    )
    ax_field.set_xlim(-0.0106, 0.0106)
    ax_field.set_ylim(-0.00035, 0.0112)
    ax_field.set_xticks((-0.01, -0.005, 0.0, 0.005, 0.01))
    ax_field.set_aspect("equal")
    ax_field.set_xlabel(r"$X_1/h$")
    ax_field.set_ylabel(r"$X_2/h$")
    ax_field.set_title("final matching-circle calculation", pad=5)
    panel_label(ax_field, "a")
    colorbar = fig.colorbar(contour, ax=ax_field, fraction=0.050, pad=0.035)
    colorbar.set_label(r"computed $Y_2/h$")

    # (b) The three windows are displayed on separate rows because they
    # overlap strongly.  They are post-processing intervals, not subdomains.
    y_positions = (3.0, 2.0, 1.0)
    for (name, lo, hi, color), y in zip(WINDOWS, y_positions):
        ax_windows.barh(
            y, hi - lo, left=lo, height=0.54,
            color=color, alpha=0.88, edgecolor="white", linewidth=0.8,
        )
        ax_windows.text(
            np.sqrt(lo * hi), y, rf"$W_{{\rm {name}}}$",
            color="white", fontsize=10.0, fontweight="bold",
            ha="center", va="center",
        )

    ax_windows.axvline(
        r_min, color="0.2", lw=1.0, ls=":",
    )
    ax_windows.axvline(
        sample_lo, color="0.45", lw=1.0, ls=(0, (3, 2)),
    )
    ax_windows.axvline(
        sample_hi, color="0.45", lw=1.0, ls=(0, (3, 2)),
    )
    ax_windows.axvline(
        matching_radius, color="0.12", lw=1.4, ls=(0, (5, 2.5)),
    )
    ax_windows.annotate(
        "", xy=(sample_lo, 3.65), xytext=(sample_hi, 3.65),
        arrowprops={"arrowstyle": "<->", "lw": 0.9, "color": "0.35"},
    )
    ax_windows.text(
        np.sqrt(sample_lo * sample_hi), 3.76,
        "retained P2 sample range", ha="center", va="bottom",
        fontsize=8.4, color="0.3",
    )
    ax_windows.text(
        r_min, 3.92, r"$r_{\min}$", ha="center", va="top",
        fontsize=8.5, rotation=90,
    )
    ax_windows.text(
        matching_radius, 3.92, r"$R_m$", ha="center", va="top",
        fontsize=8.5, rotation=90,
    )
    ax_windows.set_xscale("log")
    ax_windows.set_xlim(1.0e-6, 1.8e-2)
    ax_windows.set_ylim(0.42, 4.05)
    ax_windows.set_yticks(y_positions, ("I", "II", "III"))
    ax_windows.set_xlabel(r"reference radius $r/h$")
    ax_windows.set_ylabel("fit window")
    ax_windows.set_title(
        r"overlapping radial windows; only $\theta=0$ enters Fig. 7",
        pad=5,
    )
    ax_windows.grid(axis="x", which="major", color="0.88", lw=0.7)
    panel_label(ax_windows, "b")

    fig.subplots_adjust(left=0.105, right=0.96, bottom=0.09, top=0.95)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / "fig_esi_matching_circle.pdf"
    png = OUTPUT / "fig_esi_matching_circle.png"
    fig.savefig(
        pdf,
        metadata={
            "Creator": "make_esi_matching_circle.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {pdf} and {png}")


if __name__ == "__main__":
    make_figure()
