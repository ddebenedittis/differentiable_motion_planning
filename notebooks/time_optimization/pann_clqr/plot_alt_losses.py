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

from utils import (
    LOSS_REGISTRY,
    REPARAM_CHOICES,
    compute_trajectory_metrics,
    evaluate_continuous_cost,
    extract_trajectory_data,
    load_pickle,
    plot_cross_correlations,
    plot_density_and_changes,
    plot_training_res,
    save_training_res,
    save_timesteps_video,
)

# ============================================================================ #
# System Constants (Pannocchia) — needed for cost evaluation
# ============================================================================ #

A = np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])
B = np.array([[0.25], [2.0], [0.0]])
s0 = np.array([1.344, -4.585, 5.674])
T = 10.0
Q = 1.0 * np.eye(3)
R = 0.1 * np.eye(1)
u_max = 1.0
n_s = 3
n_u = 1


# ============================================================================ #
# Load Results
# ============================================================================ #

def load_all_results(data_dir, loss_names=None, suffix=""):
    """Load all available loss results from pickle files.

    Args:
        data_dir: directory containing pickle files
        loss_names: list of loss names to load, or None for all registered losses
        suffix: pickle name suffix (e.g. "_logsoftmax" for logsoftmax results)

    Returns:
        results: dict of {loss_name: {"sol": ..., "history": ...}}
    """
    candidates = loss_names or list(LOSS_REGISTRY.keys())
    results = {}

    for loss_name in candidates:
        try:
            sol = load_pickle(data_dir, f"sol_{loss_name}{suffix}")
            history = load_pickle(data_dir, f"history_{loss_name}{suffix}")
            results[loss_name] = {"sol": sol, "history": history}
        except (FileNotFoundError, OSError):
            pass

    return results


# ============================================================================ #
# Per-Loss Plots
# ============================================================================ #

def plot_loss_results(loss_name, result, n, results_dir, show=False):
    """Plot training results for a single loss (2x2 grid)."""
    sol = result["sol"]
    history = result["history"]

    plot_training_res(sol, history, n, sol_method=2)
    plt.suptitle(loss_name)

    if results_dir:
        save_training_res(results_dir, loss_name, sol, history, n, sol_method=2)

    if not show:
        plt.close('all')


# ============================================================================ #
# Continuous Cost Evaluation
# ============================================================================ #

def print_continuous_costs(all_results, n):
    """Evaluate and print true continuous-time costs for all losses."""
    print("=== True Continuous-Time Cost Comparison ===\n")

    for loss_name, result in all_results.items():
        sol = result["sol"]
        history = result["history"]

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

def _build_method_solutions(all_results, n):
    """Build the method_solutions dict used for density/cross-correlation plots."""
    method_solutions = {}

    for loss_name, result in all_results.items():
        method_solutions[loss_name] = {
            'sol': result['sol'],
            'history': result['history'],
            'sol_method': 2,
            'n': n,
        }

    return method_solutions


def plot_density_analysis(method_solutions, colors, results_dir, show=False):
    """Plot sampling density vs trajectory changes for all losses."""
    n_methods = len(method_solutions)
    if n_methods == 0:
        return

    n_rows = int(np.ceil(n_methods / 2))
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.5 * n_rows), squeeze=False)

    for i, (key, ms) in enumerate(method_solutions.items()):
        n_m = ms['n']
        data = extract_trajectory_data(ms, n_m)
        metrics = compute_trajectory_metrics(data, n_m, T)
        plot_density_and_changes(data, metrics, key, colors, axes=axs[i // 2, i % 2])

    # Remove unused axes
    for j in range(i + 1, n_rows * 2):
        fig.delaxes(axs[j // 2, j % 2])

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "density_analysis.pdf"), bbox_inches='tight')

    if not show:
        plt.close(fig)


def plot_cross_correlation_analysis(method_solutions, colors, results_dir, show=False):
    """Plot cross-correlation for all losses."""
    for key, ms in method_solutions.items():
        n_m = ms['n']
        data = extract_trajectory_data(ms, n_m)
        metrics = compute_trajectory_metrics(data, n_m, T)
        fig = plot_cross_correlations(data, metrics, key, colors, max_lag=30)

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            safe_key = key.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
            fig.savefig(
                os.path.join(results_dir, f"cross_corr_{safe_key}.pdf"),
                bbox_inches='tight',
            )

        if not show:
            plt.close(fig)


def plot_comparison(all_results, results_dir, show=False):
    """Comparison plot: final timesteps for all losses."""
    if len(all_results) <= 1:
        return

    n_losses = len(all_results)
    fig, axes = plt.subplots(1, n_losses, figsize=(3.2 * n_losses, 3.2))
    if n_losses == 1:
        axes = [axes]

    for ax, (loss_name, result) in zip(axes, all_results.items()):
        history = result["history"]
        dts_final = history[-1]['dts'].flatten()
        times = np.cumsum(dts_final)
        ax.plot(times, dts_final)
        ax.set_xlabel("Time")
        ax.set_ylabel("dt")
        ax.set_title(loss_name)

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "comparison.pdf"), bbox_inches='tight')

    if not show:
        plt.close(fig)


def save_summary(all_results, n, results_dir):
    """Compute and save metrics summary JSON."""
    summary = {}

    for loss_name, result in all_results.items():
        sol = result["sol"]
        history = result["history"]

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
    parser.add_argument("--n", type=int, default=160, help="Number of timesteps")
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
        all_results = load_all_results(data_dir, loss_names=args.loss,
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
                plot_loss_results(loss_name, result, args.n, results_dir, show=args.show)

            # Continuous costs
            print_continuous_costs(all_results, args.n)

            # Comparison plot
            plot_comparison(all_results, results_dir, show=args.show)

        # Cross-loss analysis
        method_solutions = _build_method_solutions(all_results, args.n)
        if method_solutions:
            print(f"Analysis: {len(method_solutions)} losses")
            plot_density_analysis(method_solutions, colors, results_dir, show=args.show)
            plot_cross_correlation_analysis(method_solutions, colors, results_dir, show=args.show)

        # Summary
        save_summary(all_results, args.n, results_dir)

        # Timestep evolution videos
        if args.save_video:
            print("Saving timestep evolution videos...")
            _save_all_videos(all_results, data_dir, results_dir, suffix=suffix)

        print(f"\nPlots saved to: {results_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
