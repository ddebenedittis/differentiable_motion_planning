#!/usr/bin/env python
"""Compare wall-clock times of the differentiable (lsqr) method vs OCSLC.

Reads:
    csv/stiff_sys_solve_times_sweep__lsqr.csv  (ours)
    csv/ocslc_stiff_sys_solve_times_sweep.csv  (ocslc)

Saves six PDFs to ../results/stiff_sys/:
    compare_{setup,solve,total}_abs.pdf    -- absolute times (log-y)
    compare_{setup,solve,total}_ratio.pdf  -- speedup factor ocslc / ours

Mapping between the two timing breakdowns:
    setup   <-  dQP_create            (ours)   vs  precompute + setup  (ocslc)
    solve   <-  train_total           (ours)   vs  solve               (ocslc)
    total   <-  total                 (ours)   vs  total               (ocslc)
"""

import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimp.utils import init_matplotlib, get_colors  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
OURS_CSV = os.path.join(CSV_DIR, "stiff_sys_solve_times_sweep__lsqr.csv")
OCSLC_CSV = os.path.join(CSV_DIR, "ocslc_stiff_sys_solve_times_sweep.csv")

RESULTS_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "results", "stiff_sys"
)

# (label, ours_column, ocslc_column, output_stem)
COMPONENTS = [
    ("Setup",  "setup_ours",  "setup_ocslc",  "setup"),
    ("Solve",  "train_total", "solve",        "solve"),
    ("Total",  "total",       "total_ocslc",  "total"),
]


def load_data():
    ours = pd.read_csv(OURS_CSV).rename(columns={"n_control": "horizon"})
    ocslc = pd.read_csv(OCSLC_CSV).rename(columns={"n_steps": "horizon"})

    # Synthesize matching column names for the merge.
    ours = ours.assign(setup_ours=ours["dQP_create"])
    ocslc = ocslc.assign(
        setup_ocslc=ocslc["precompute"] + ocslc["setup"],
        total_ocslc=ocslc["total"],
    )

    keep_ours = ["n_states", "horizon", "setup_ours", "train_total", "total"]
    keep_ocslc = ["n_states", "horizon", "setup_ocslc", "solve", "total_ocslc"]
    merged = pd.merge(
        ours[keep_ours],
        ocslc[keep_ocslc],
        on=["n_states", "horizon"],
        how="inner",
    ).sort_values(["n_states", "horizon"])

    if merged.empty:
        raise RuntimeError(
            "No overlapping (n_states, horizon) cells between the two CSVs."
        )
    return merged


def plot_absolute(df, label, ours_col, ocslc_col, *, figsize=(3.8, 2.8)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    colors = get_colors("matlab")
    n_states_vals = sorted(df["n_states"].unique())

    for i, n_states in enumerate(n_states_vals):
        sub = df[df["n_states"] == n_states].sort_values("horizon")
        c = colors[i % len(colors)]
        ax.plot(
            sub["horizon"], sub[ocslc_col],
            linestyle="--", marker="o", markerfacecolor="none",
            color=c, linewidth=1.3, markersize=4,
        )
        ax.plot(
            sub["horizon"], sub[ours_col],
            linestyle="-", marker="s",
            color=c, linewidth=1.5, markersize=4,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Horizon $N$")
    ax.set_ylabel(f"{label} time [s]")

    color_handles = [
        mlines.Line2D([], [], color=colors[i % len(colors)], linewidth=2,
                      label=f"$n_s={int(ns)}$")
        for i, ns in enumerate(n_states_vals)
    ]
    method_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", marker="s",
                      markersize=4, label="Ours"),
        mlines.Line2D([], [], color="black", linestyle="--", marker="o",
                      markerfacecolor="none", markersize=4, label="OCSLC"),
    ]
    leg1 = ax.legend(handles=color_handles, fontsize=7, ncol=2,
                     loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, fontsize=7,
              loc="lower right", framealpha=0.9)
    return fig


def plot_ratio(df, label, ours_col, ocslc_col, *, figsize=(3.8, 2.8)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    colors = get_colors("matlab")
    n_states_vals = sorted(df["n_states"].unique())

    for i, n_states in enumerate(n_states_vals):
        sub = df[df["n_states"] == n_states].sort_values("horizon")
        ratio = sub[ocslc_col] / sub[ours_col]
        ax.plot(
            sub["horizon"], ratio, "s-",
            color=colors[i % len(colors)],
            linewidth=1.5, markersize=4,
            label=f"$n_s={int(n_states)}$",
        )

    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon $N$")
    ax.set_ylabel(f"{label} speedup (OCSLC / Ours)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    return fig


def main():
    init_matplotlib("matlab")
    df = load_data()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Overlapping cells: {len(df)}")
    print(f"  n_states: {sorted(df['n_states'].unique())}")
    print(f"  horizon : {sorted(df['horizon'].unique())}")

    for label, ours_col, ocslc_col, stem in COMPONENTS:
        fig_abs = plot_absolute(df, label, ours_col, ocslc_col)
        fig_abs.savefig(
            os.path.join(RESULTS_DIR, f"compare_{stem}_abs.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig_abs)

        fig_ratio = plot_ratio(df, label, ours_col, ocslc_col)
        fig_ratio.savefig(
            os.path.join(RESULTS_DIR, f"compare_{stem}_ratio.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig_ratio)

        med = (df[ocslc_col] / df[ours_col]).median()
        print(f"  {label:<6} median speedup: {med:6.1f}x")

    print(f"\nSaved 6 figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
