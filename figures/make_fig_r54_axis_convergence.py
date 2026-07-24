"""Make the manuscript figure for the matching-circle r^(5/4) FEM test.

Chart contract
--------------
Question
    Does the physical pure-shear strip realize the nuisance-free exact-axis
    r^(5/4) residual and its parameter-free amplitude under core and angular
    refinement?
Reader
    Soft-matter and fracture-mechanics readers assessing the numerical
    evidence behind the asymptotic result.
Evidence
    The retained matching-circle profile, the three overlapping exact-axis
    radial fit windows, and their four-case core/angular refinement sequence.
Takeaway
    Both the freely fitted exponent and the amplitude ratio approach their
    analytical values. Matching-radius and free-two-power sensitivities are
    reported in the caption rather than represented as statistical error
    bars.

This script reads the compact, hashed campaign summary and the final retained
P2 profile. It does not run FEM or refit any field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY = (
    ROOT / "data" / "fem" / "global_local"
    / "global_local_campaign_summary_2026-07-23.json"
)
PROFILE = (
    ROOT / "data" / "fem" / "global_local"
    / "global_local_profile_lam16_core2p5e6_nt120_rm1e2.npz"
)
FIG = HERE / "rendered"

CASE_ORDER = (
    "core_1e-5_ntheta60",
    "core_5e-6_ntheta60",
    "core_2.5e-6_ntheta60",
    "core_2.5e-6_ntheta120",
)

SERIES = (
    (r"I: $[1.5{\times}10^{-4},\,1.6{\times}10^{-3}]$",
     "#666666", "o", (0, (4, 2))),
    (r"II: $[3.0{\times}10^{-4},\,3.0{\times}10^{-3}]$",
     "#0072B2", "s", (0, (5, 1.5, 1.5, 1.5))),
    (r"III: $[6.0{\times}10^{-4},\,3.8{\times}10^{-3}]$",
     "#c1272d", "^", "-"),
)

WINDOWS = (
    ("I", 1.5e-4, 1.6e-3, "#666666"),
    ("II", 3.0e-4, 3.0e-3, "#0072B2"),
    ("III", 6.0e-4, 3.8e-3, "#c1272d"),
)


def panel_label(ax, letter: str) -> None:
    ax.text(
        0.025, 0.97, letter, transform=ax.transAxes,
        fontsize=12.5, fontweight="bold", va="top",
    )


def load_campaign(summary: Path) -> tuple[dict, list[dict]]:
    data = json.loads(summary.read_text())
    cases = [data["core_and_angular_cases"][key] for key in CASE_ORDER]

    reference_windows = cases[0]["estimator"]["windows"]
    for case in cases:
        if case["estimator"]["windows"] != reference_windows:
            raise ValueError("core cases do not share the declared windows")
    return data, cases


def load_profile(profile: Path) -> dict[str, np.ndarray | float]:
    with np.load(profile) as archive:
        return {
            "X": np.asarray(archive["X"], dtype=float),
            "Y": np.asarray(archive["Y"], dtype=float),
            "Y2": np.asarray(archive["Y2"], dtype=float),
            "r_min": float(archive["r_min"]),
            "matching_radius": float(archive["matching_radius"]),
            "sample_lo": float(np.min(archive["r"])),
            "sample_hi": float(np.max(archive["r"])),
        }


def make_figure(
        summary: Path = SUMMARY,
        profile_path: Path = PROFILE,
        output_dir: Path = FIG) -> None:
    data, cases = load_campaign(summary)
    source_files = data["core_and_angular_cases"][
        "core_2.5e-6_ntheta120"
    ]["source_files"]
    if profile_path.name != source_files["profile"]:
        raise ValueError(
            "profile filename does not match the final case recorded "
            "in the campaign summary"
        )
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if profile_hash != source_files["profile_sha256"]:
        raise ValueError(
            "profile SHA-256 does not match the campaign summary"
        )
    profile = load_profile(profile_path)
    q = np.array([case["estimator"]["axis_q"] for case in cases])
    amplitude_ratio = 1.0 + np.array([
        case["estimator"]["axis_amplitude_relative_error"]
        for case in cases
    ])

    labels = []
    for case_index, case in enumerate(cases, start=1):
        solver = case["solver"]
        core = solver["r_min"]
        core_label = {
            1.0e-5: r"10^{-5}",
            5.0e-6: r"5{\times}10^{-6}",
            2.5e-6: r"2.5{\times}10^{-6}",
        }[core]
        labels.append(
            rf"case {case_index}" + "\n"
            + rf"$({core_label},\,{solver['n_theta']})$"
        )

    plt.rcParams.update({
        "font.size": 9.2,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "lines.linewidth": 1.65,
        "lines.markersize": 5.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    x = np.arange(len(cases), dtype=float)
    fig = plt.figure(figsize=(7.15, 5.35))
    outer_grid = fig.add_gridspec(
        2, 1, height_ratios=(0.98, 1.0),
        left=0.085, right=0.985, bottom=0.165, top=0.965,
        hspace=0.58,
    )
    top_grid = outer_grid[0].subgridspec(
        1, 2, width_ratios=(0.45, 0.55), wspace=0.34,
    )
    bottom_grid = outer_grid[1].subgridspec(1, 2, wspace=0.34)
    ax_field = fig.add_subplot(top_grid[0, 0])
    ax_windows = fig.add_subplot(top_grid[0, 1])
    ax_q = fig.add_subplot(bottom_grid[0, 0])
    ax_a = fig.add_subplot(bottom_grid[0, 1], sharex=ax_q)

    # (a) Direct displacement field from the retained exact-inner-restriction
    # profile. The dashed curve is the internal matching interface, not a hole.
    X = profile["X"]
    Y = profile["Y"]
    Y2 = profile["Y2"]
    r_min = float(profile["r_min"])
    matching_radius = float(profile["matching_radius"])
    sample_lo = float(profile["sample_lo"])
    sample_hi = float(profile["sample_hi"])
    levels = np.linspace(float(np.nanmin(Y2)), float(np.nanmax(Y2)), 17)
    contour = ax_field.contourf(
        X, Y, Y2, levels=levels, cmap="cividis", extend="both",
    )
    theta = np.linspace(0.0, np.pi, 500)
    ax_field.plot(
        matching_radius * np.cos(theta),
        matching_radius * np.sin(theta),
        color="0.12", lw=1.35, ls=(0, (5, 2.5)),
    )
    ax_field.plot(
        [sample_lo, sample_hi], [0.0, 0.0],
        color="#E69F00", lw=1.9,
    )
    ax_field.plot(
        [-sample_hi, -sample_lo], [0.0, 0.0],
        color="0.15", lw=1.15,
    )
    ax_field.plot(
        sample_hi * np.cos(theta), sample_hi * np.sin(theta),
        color="0.45", lw=0.9, ls=":",
    )
    ax_field.text(
        0.0052, 0.00955, r"$R_m/h=0.01$",
        fontsize=8.0, color="0.15", ha="center",
    )
    ax_field.text(
        0.0016, 0.00058, r"fit axis $\theta=0$",
        color="#9A6700", fontsize=7.5, ha="left",
    )
    ax_field.text(
        -0.0092, 0.00058, "crack face",
        color="0.15", fontsize=7.5, ha="left",
    )
    ax_field.set_xlim(-0.0105, 0.0105)
    ax_field.set_ylim(-0.00035, 0.01065)
    ax_field.set_xticks((-0.01, 0.0, 0.01))
    ax_field.set_yticks((0.0, 0.005, 0.01))
    ax_field.set_aspect("equal")
    ax_field.set_xlabel(r"$X_1/h$")
    ax_field.set_ylabel(r"$X_2/h$")
    ax_field.set_title("matching-circle field", pad=3)
    panel_label(ax_field, "a")
    colorbar_ticks = np.linspace(
        float(np.nanmin(Y2)), float(np.nanmax(Y2)), 4,
    )
    if abs(colorbar_ticks[0]) < 5.0e-8:
        colorbar_ticks[0] = 0.0
    colorbar = fig.colorbar(
        contour, ax=ax_field, fraction=0.045, pad=0.025,
        ticks=colorbar_ticks,
        format=FuncFormatter(
            lambda value, _: "0.00" if abs(value) < 5.0e-3
            else f"{value:.2f}"
        ),
    )
    colorbar.ax.tick_params(labelsize=7.1)
    colorbar.ax.set_title(r"$Y_2/h$", fontsize=8.5, pad=2)

    # (b) The windows overlap. They are fitting intervals applied to the same
    # exact-axis trace, not separate numerical domains.
    y_positions = (3.0, 2.0, 1.0)
    for (name, lo, hi, color), y_position in zip(WINDOWS, y_positions):
        ax_windows.barh(
            y_position, hi - lo, left=lo, height=0.50,
            color=color, alpha=0.9, edgecolor="white", linewidth=0.7,
        )
        ax_windows.text(
            np.sqrt(lo * hi), y_position, rf"$W_{{\rm {name}}}$",
            color="white", fontsize=8.7, fontweight="bold",
            ha="center", va="center",
        )
    ax_windows.axvspan(
        sample_lo, sample_hi, color="0.75", alpha=0.18, zorder=0,
    )
    ax_windows.axvline(r_min, color="0.2", lw=0.9, ls=":")
    ax_windows.axvline(
        sample_lo, color="0.45", lw=0.9, ls=(0, (3, 2)),
    )
    ax_windows.axvline(
        sample_hi, color="0.45", lw=0.9, ls=(0, (3, 2)),
    )
    ax_windows.axvline(
        matching_radius, color="0.12", lw=1.25, ls=(0, (5, 2.5)),
    )
    ax_windows.text(
        np.sqrt(sample_lo * sample_hi), 3.63, "sampled radii",
        ha="center", va="center", fontsize=7.4, color="0.3",
    )
    ax_windows.text(
        r_min, 0.55, r"$r_{\min}$", ha="center", va="bottom",
        fontsize=7.5, rotation=90,
    )
    ax_windows.text(
        matching_radius, 3.82, r"$R_m$", ha="center", va="top",
        fontsize=7.5, rotation=90,
    )
    ax_windows.set_xscale("log")
    ax_windows.set_xlim(1.5e-6, 1.45e-2)
    ax_windows.set_ylim(0.45, 3.9)
    ax_windows.set_yticks(y_positions, ("I", "II", "III"))
    ax_windows.set_xlabel(r"reference radius $r/h$")
    ax_windows.set_title("overlapping radial fit windows", pad=3)
    ax_windows.grid(axis="x", which="major", color="0.88", lw=0.65)
    panel_label(ax_windows, "b")

    for index, (label, color, marker, linestyle) in enumerate(SERIES):
        ax_q.plot(
            x, q[:, index], label=label, color=color, marker=marker,
            linestyle=linestyle, markerfacecolor="white",
            markeredgewidth=1.4,
        )
        ax_a.plot(
            x, amplitude_ratio[:, index], color=color, marker=marker,
            linestyle=linestyle, markerfacecolor="white",
            markeredgewidth=1.4,
        )

    ax_q.axhline(1.25, color="0.15", lw=1.2, ls="--", zorder=0)
    ax_a.axhline(1.0, color="0.15", lw=1.2, ls="--", zorder=0)
    ax_q.text(
        0.04, 0.055, r"$q=5/4$", transform=ax_q.transAxes,
        ha="left", va="bottom", color="0.15", fontsize=8.5,
    )
    ax_a.text(
        0.04, 0.055,
        r"$A_{\rm ax}/A_{\rm ax,pred}=1$",
        transform=ax_a.transAxes,
        ha="left", va="bottom", color="0.15", fontsize=8.5,
    )

    for ax in (ax_q, ax_a):
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.18, 3.18)
        ax.grid(axis="y", color="0.88", lw=0.7)
        ax.tick_params(direction="out")

    ax_q.set_ylim(1.2475, 1.2790)
    ax_a.set_ylim(0.99, 1.235)
    ax_q.set_ylabel(r"fitted exact-axis exponent $q$")
    ax_a.set_ylabel(
        r"amplitude ratio $A_{\rm ax}/A_{\rm ax,pred}$"
    )
    panel_label(ax_q, "c")
    panel_label(ax_a, "d")

    handles, _ = ax_q.get_legend_handles_labels()
    fig.legend(
        handles, (r"$W_{\rm I}$", r"$W_{\rm II}$", r"$W_{\rm III}$"),
        loc="center", ncol=3,
        bbox_to_anchor=(0.535, 0.505), frameon=False,
        handlelength=2.0, columnspacing=0.95, fontsize=8.1,
        title=r"fit windows used in panels (c,d)",
        title_fontsize=8.1,
    )
    fig.text(
        0.535, 0.038,
        r"global strip case $(r_{\min}/h,\,n_\theta)$",
        ha="center", va="center", fontsize=9.8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "fig_r54_axis_convergence.pdf",
        metadata={
            "Creator": "make_fig_r54_axis_convergence.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_dir / "fig_r54_axis_convergence.png")
    plt.close(fig)
    print(
        "wrote "
        f"{output_dir}/fig_r54_axis_convergence.{{pdf,png}} "
        f"from {summary} and {profile_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=SUMMARY,
        help="curated or freshly regenerated campaign summary JSON",
    )
    parser.add_argument(
        "--profile", type=Path, default=PROFILE,
        help="retained final matching-circle P2 profile",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIG,
        help="directory for the PDF and PNG outputs",
    )
    args = parser.parse_args()
    make_figure(args.summary, args.profile, args.output_dir)
