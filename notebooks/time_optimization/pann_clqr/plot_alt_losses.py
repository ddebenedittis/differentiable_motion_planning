#!/usr/bin/env python
"""Visualization script for alternative loss function experiments.

Loads pickles from data/alt_losses/ and produces plots in results/alt_losses/.

Usage:
    python plot_alt_losses.py
    python plot_alt_losses.py --loss L_IV L_FI
    python plot_alt_losses.py --analysis-only
    python plot_alt_losses.py --show
    python plot_alt_losses.py --save-video
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from pann_clqr import A, B, s0, T, Q, R, u_max, n_s, n_u
from utils import (
    LOSS_REGISTRY,
    REPARAM_CHOICES,
    load_loss_results,
    plot_loss_results,
    plot_loss_comparison,
    plot_density_analysis_grid,
    evaluate_continuous_cost,
    load_pickle,
    plot_training_res,
    save_training_res,
    extract_trajectory_data,
    compute_trajectory_metrics,
    plot_density_and_changes,
    save_timesteps_video,
)


# ============================================================================ #
# Continuous Cost Evaluation
# ============================================================================ #

def print_continuous_costs(all_results):
    """Evaluate and print true continuous-time costs for all losses."""
    print("=== True Continuous-Time Cost Comparison ===\n")

    for loss_name, result in all_results.items():
        sol = result["sol"]
        history = result["history"]
        n = result["n"]

        dts_final = history[-1]['dts']
        inputs_qp = [sol[n + k].detach().float() for k in range(n)]

        try:
            true_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, s0, A, B, Q, R, T,
            )
        except Exception as e:
            print(f"{loss_name}: cost evaluation failed: {e}")
            continue

        print(f"{loss_name}:")
        print(f"  Training loss (final):  {history[-1]['loss']:.4f}")
        print(f"  Loss OCP (final):       {history[-1]['loss_ocp']:.4f}")
        print(f"  Loss reg (final):       {history[-1]['loss_reg']:.4f}")
        print(f"  Lambda hat (final):     {history[-1]['lambda_hat']:.4f}")
        print(f"  True continuous cost:   {true_cost:.4f}")
        if isinstance(dts_final, np.ndarray):
            print(f"  dt range: [{np.min(dts_final):.5f}, {np.max(dts_final):.5f}]")
            print(f"  dt std:   {np.std(dts_final):.5f}")
        print()


# ============================================================================ #
# Cross-Loss Analysis
# ============================================================================ #

def _build_method_solutions(all_results):
    """Build the method_solutions dict used for density/cross-correlation plots."""
    method_solutions = {}

    for loss_name, result in all_results.items():
        method_solutions[loss_name] = {
            'sol': result['sol'],
            'history': result['history'],
            'sol_method': 2,
            'n': result['n'],
        }

    return method_solutions


def save_summary(all_results, results_dir):
    """Compute and save metrics summary JSON."""
    summary = {}

    for loss_name, result in all_results.items():
        sol = result["sol"]
        history = result["history"]
        n = result["n"]

        dts_final = history[-1]['dts']
        inputs_qp = [sol[n + k].detach().float() for k in range(n)]

        try:
            cont_cost = evaluate_continuous_cost(inputs_qp, dts_final, s0, A, B, Q, R, T)
        except Exception:
            cont_cost = None

        summary[loss_name] = {
            "continuous_cost": cont_cost,
            "final_loss": history[-1]['loss'],
            "final_loss_ocp": history[-1]['loss_ocp'],
            "final_loss_reg": history[-1]['loss_reg'],
            "final_lambda_hat": history[-1]['lambda_hat'],
        }

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for loss_name, metrics in summary.items():
        print(f"\n{loss_name}:")
        cost_str = f"{metrics['continuous_cost']:.6f}" if metrics['continuous_cost'] is not None else "N/A"
        print(f"  Continuous cost:  {cost_str}")
        print(f"  Final loss (ocp): {metrics['final_loss_ocp']:.6f}")
        print(f"  Final loss (reg): {metrics['final_loss_reg']:.6f}")
        print(f"  Final lambda:     {metrics['final_lambda_hat']:.6f}")

    return summary


# ============================================================================ #
# Timestep Evolution Videos
# ============================================================================ #

def _save_all_videos(all_results, data_dir, results_dir, suffix=""):
    """Generate timestep evolution videos from saved dts distribution pickles."""
    for loss_name in all_results:
        try:
            dts_dist = load_pickle(data_dir, f"dts_dist_{loss_name}{suffix}")
        except (FileNotFoundError, OSError):
            print(f"  No dts distribution for {loss_name}, skipping video")
            continue

        print(f"  Video: {loss_name}")
        save_timesteps_video(results_dir, f"{loss_name}_timesteps", dts_all=dts_dist)


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Visualization for alternative loss function experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle source directory (default: data/alt_losses)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Plot output directory (default: results/alt_losses)",
    )
    parser.add_argument(
        "--loss", nargs="+", default=None,
        help="Which losses to plot (default: all available)",
    )
    parser.add_argument("--analysis-only", action="store_true", help="Only cross-loss analysis plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--save-video", action="store_true", help="Save timestep evolution videos")
    parser.add_argument(
        "--reparam", default="both",
        choices=[*REPARAM_CHOICES, "both"],
        help="Which reparametrization results to plot (default: both)",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "alt_losses")
    results_dir_base = args.results_dir or os.path.join(script_dir, "results", "alt_losses")

    # Try to use project matplotlib styling
    try:
        from dimp.utils import init_matplotlib, get_colors
        init_matplotlib()
        colors = get_colors()
    except ImportError:
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#D55E00']

    reparams = list(REPARAM_CHOICES) if args.reparam == "both" else [args.reparam]

    for reparam in reparams:
        suffix = f"_{reparam}" if reparam != "softmax" else ""
        results_dir = (os.path.join(results_dir_base, reparam)
                       if reparam != "softmax" else results_dir_base)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'#' * 50}")
        print(f"# Reparametrization: {reparam}")
        print(f"{'#' * 50}")
        print(f"Loading data from: {data_dir}")
        all_results = load_loss_results(data_dir, loss_names=args.loss or "all",
                                       suffix=suffix)

        if not all_results:
            print(f"No {reparam} results found. Skipping.")
            continue

        loaded = list(all_results.keys())
        print(f"Loaded losses: {loaded}")

        if not args.analysis_only:
            # Per-loss plots
            for loss_name, result in all_results.items():
                print(f"Plotting {loss_name}...")
                plot_loss_results(loss_name, result, results_dir, show=args.show)

            # Continuous costs
            print_continuous_costs(all_results)

            # Comparison plot
            plot_loss_comparison(all_results, results_dir, show=args.show, filename="comparison")

        # Cross-loss analysis
        method_solutions = _build_method_solutions(all_results)
        if method_solutions:
            print(f"Analysis: {len(method_solutions)} losses")
            plot_density_analysis_grid(method_solutions, T, colors, results_dir, show=args.show)

        # Summary
        save_summary(all_results, results_dir)

        # Timestep evolution videos
        if args.save_video:
            print("Saving timestep evolution videos...")
            _save_all_videos(all_results, data_dir, results_dir, suffix=suffix)

        print(f"\nPlots saved to: {results_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
