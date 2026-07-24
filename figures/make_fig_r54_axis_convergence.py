"""Make the manuscript figure for the exact-axis r^(5/4) FEM test.

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
    The three common radial windows from the curated global--local campaign,
    plotted through the core/angular refinement sequence.
Takeaway
    Both the freely fitted exponent and the amplitude ratio approach their
    analytical values. Matching-radius and free-two-power sensitivities are
    reported in the caption rather than represented as statistical error
    bars.

This script reads only the compact, hashed campaign summary. It does not run
FEM or refit any field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY = (
    ROOT / "data" / "fem" / "global_local"
    / "global_local_campaign_summary_2026-07-23.json"
)
FIG = HERE / "rendered"

CASE_ORDER = (
    "core_1e-5_ntheta60",
    "core_5e-6_ntheta60",
    "core_2.5e-6_ntheta60",
    "core_2.5e-6_ntheta120",
)

SERIES = (
    ("inner annulus", "#666666", "o", (0, (4, 2))),
    ("middle annulus", "#0072B2", "s", (0, (5, 1.5, 1.5, 1.5))),
    ("outer annulus", "#c1272d", "^", "-"),
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


def make_figure(
        summary: Path = SUMMARY,
        output_dir: Path = FIG) -> None:
    data, cases = load_campaign(summary)
    q = np.array([case["estimator"]["axis_q"] for case in cases])
    amplitude_ratio = 1.0 + np.array([
        case["estimator"]["axis_amplitude_relative_error"]
        for case in cases
    ])

    labels = []
    for case in cases:
        solver = case["solver"]
        core = solver["r_min"]
        core_label = {
            1.0e-5: r"$10^{-5}$",
            5.0e-6: r"$5{\times}10^{-6}$",
            2.5e-6: r"$2.5{\times}10^{-6}$",
        }[core]
        labels.append(
            core_label + "\n" + rf"$n_\theta={solver['n_theta']}$"
        )

    plt.rcParams.update({
        "font.size": 11.5,
        "axes.labelsize": 12.5,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    x = np.arange(len(cases), dtype=float)
    fig, (ax_q, ax_a) = plt.subplots(
        1, 2, figsize=(7.15, 3.25), sharex=True,
        gridspec_kw={"wspace": 0.27},
    )

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
        ha="left", va="bottom", color="0.15",
    )
    ax_a.text(
        0.04, 0.055,
        r"$A_{\rm ax}/A_{\rm ax,pred}=1$",
        transform=ax_a.transAxes,
        ha="left", va="bottom", color="0.15",
    )

    matching_spans = data["derived_robustness_checks"][
        "matching_radius_axis_q_span_by_window"
    ]
    ax_q.text(
        0.98, 0.94,
        "$R_m/h=0.005$--$0.02$\n"
        rf"max. $\Delta q={max(matching_spans):.2e}$",
        transform=ax_q.transAxes, ha="right", va="top",
        fontsize=9.5, color="0.25",
    )

    for ax in (ax_q, ax_a):
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.18, 3.18)
        ax.grid(axis="y", color="0.88", lw=0.7)
        ax.tick_params(direction="out")
        ax.set_xlabel(r"$r_{\min}/h$ and angular sectors")

    ax_q.set_ylim(1.2475, 1.2790)
    ax_a.set_ylim(0.99, 1.235)
    ax_q.set_ylabel(r"fitted exact-axis exponent $q$")
    ax_a.set_ylabel(
        r"amplitude ratio $A_{\rm ax}/A_{\rm ax,pred}$"
    )
    panel_label(ax_q, "a")
    panel_label(ax_a, "b")

    handles, legend_labels = ax_q.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.015), frameon=False,
        handlelength=2.8, columnspacing=1.5,
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.105, right=0.985)

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
        f"from {summary}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=SUMMARY,
        help="curated or freshly regenerated campaign summary JSON",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIG,
        help="directory for the PDF and PNG outputs",
    )
    args = parser.parse_args()
    make_figure(args.summary, args.output_dir)
