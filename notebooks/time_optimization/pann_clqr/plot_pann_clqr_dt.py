#!/usr/bin/env python
"""Visualization script for differentiable time optimization experiments.

Loads pickles from data/pann_clqr_dt/ and produces plots in results/pann_clqr_dt/.

Usage:
    python plot_pann_clqr_dt.py
    python plot_pann_clqr_dt.py --method rep zoh
    python plot_pann_clqr_dt.py --analysis-only
    python plot_pann_clqr_dt.py --show
    python plot_pann_clqr_dt.py --save-video
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from pann_clqr import (
    create_pann_clqr,
    A, B, s0, T, Q, R, u_max, n_s, n_u,
)
from utils import (
    REPARAM_CHOICES,
    pickle_name,
    load_pickle,
    evaluate_continuous_cost,
    plot_training_res,
    save_training_res,
    extract_trajectory_data,
    compute_trajectory_metrics,
    plot_density_and_changes,
    save_timesteps_video,
)


# ============================================================================ #
# Method Configuration (used by load_all_results)
# ============================================================================ #

@dataclass(frozen=True)
class MethodConfig:
    key: str                           # "aux", "rep", "hs", "zoh"
    sol_pickle: str                    # base pickle name for solution
    history_pickle: str                # base pickle name for history
    sol_method: int                    # 1 (aux) or 2 (rep/zoh)
    internal_methods: tuple[str, ...]  # sub-methods within this result
    uses_custom_n: bool = False        # True for zoh (needs pickle_name suffix)


METHOD_CONFIGS: list[MethodConfig] = [
    MethodConfig("aux", "sol_aux", "history_aux", 1, ("time scaled", "time bar scaled")),
    MethodConfig("rep", "sol_rep", "history_rep", 2, ("time scaled",)),
    MethodConfig("hs", "sol_hs", "history_hs", 2, ("uniform_resample", "substeps")),
    MethodConfig("zoh", "sol_zoh", "history_zoh", 2, ("exact_zoh_integrated",), uses_custom_n=True),
]


N = 1000


# ============================================================================ #
# Load Results
# ============================================================================ #

def load_all_results(data_dir, n=160, n_zoh=80, suffix=""):
    """Load all available method results from pickle files.

    Args:
        suffix: pickle name suffix (e.g. "_logsoftmax" for logsoftmax results)

    Returns:
        results: dict of {method_name: {sol, history, sol_method, n, internal_methods}}
    """
    results = {}

    for cfg in METHOD_CONFIGS:
        try:
            if cfg.uses_custom_n:
                sol_pkl = pickle_name(cfg.sol_pickle + suffix, n_zoh, n)
                hist_pkl = pickle_name(cfg.history_pickle + suffix, n_zoh, n)
                n_method = n_zoh
            else:
                sol_pkl = cfg.sol_pickle + suffix
                hist_pkl = cfg.history_pickle + suffix
                n_method = n

            sol = load_pickle(data_dir, sol_pkl)
            history = load_pickle(data_dir, hist_pkl)
            results[cfg.key] = {
                "sol": sol,
                "history": history,
                "sol_method": cfg.sol_method,
                "n": n_method,
                "internal_methods": list(cfg.internal_methods),
            }
        except (FileNotFoundError, OSError):
            pass

    return results


# ============================================================================ #
# Per-Method Plots (cells 14, 19, 25, 31, 33, 37)
# ============================================================================ #

def plot_method_results(name, result, results_dir, show=False):
    """Plot training results for a single method (2x3 grid per sub-method)."""
    sol_dict = result["sol"]
    history = result["history"]
    n_method = result["n"]
    sol_method = result["sol_method"]
    internal_methods = result["internal_methods"]

    for method in internal_methods:
        sol = sol_dict[method]
        hist_m = [h for h in history if h['method'] == method]
        if not hist_m:
            print(f"  No history for {name}/{method}, skipping")
            continue

        plot_training_res(sol, hist_m, n_method, sol_method=sol_method)
        plt.suptitle(f"{name}: {method}")

        if results_dir:
            _key = method.replace(' ', '_')
            save_training_res(results_dir, f"{name}_{_key}", sol, hist_m, n_method, sol_method)

        if not show:
            plt.close('all')


# ============================================================================ #
# HS Comparison (cell 26)
# ============================================================================ #

def plot_hs_comparison(result, n, results_dir, show=False):
    """Compare uniform resampling vs substeps (loss, dt distributions, histograms)."""
    sol_hs = result["sol"]
    history_hs = result["history"]
    methods = result["internal_methods"]

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4), constrained_layout=True)

    for method in methods:
        history_method = [h for h in history_hs if h['method'] == method]
        if not history_method:
            continue
        axs[0, 0].plot([h['loss'] for h in history_method], label=method)
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title("Loss Convergence")
    axs[0, 0].legend()

    for method in methods:
        history_method = [h for h in history_hs if h['method'] == method]
        if not history_method:
            continue
        d_arr = history_method[-1]['dts']
        times = np.cumsum(d_arr)
        axs[0, 1].plot(times, d_arr, label=method)
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Timestep duration")
    axs[0, 1].set_title("Final Timestep Distributions")
    axs[0, 1].legend()

    for i, method in enumerate(methods):
        if i >= 2:
            break
        history_method = [h for h in history_hs if h['method'] == method]
        if not history_method:
            continue
        d_arr = history_method[-1]['dts']
        axs[1, i].hist(d_arr.flatten(), bins=30, alpha=0.7, edgecolor='black')
        axs[1, i].set_xlabel("Timestep duration")
        axs[1, i].set_ylabel("Count")
        axs[1, i].set_title(f"Histogram: {method}")
        axs[1, i].axvline(T / n, color='r', linestyle='--', label=f"uniform={T/n:.4f}")
        axs[1, i].legend()

    fig.suptitle("Comparison: Uniform Resampling vs Substeps", fontsize=14)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "hs_comparison.pdf"), bbox_inches='tight')

    if not show:
        plt.close(fig)


# ============================================================================ #
# Continuous Cost Evaluation (cell 27)
# ============================================================================ #

def print_continuous_costs(all_results, n):
    """Evaluate and print true continuous-time costs for all methods."""
    print("=== True Continuous-Time Cost Comparison ===\n")

    for name, result in all_results.items():
        sol_dict = result["sol"]
        history = result["history"]
        n_method = result["n"]

        for method in result["internal_methods"]:
            sol = sol_dict[method]
            hist_m = [h for h in history if h['method'] == method]
            if not hist_m:
                continue

            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_method + i].detach().float() for i in range(n_method)]

            try:
                true_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, s0, A, B, Q, R, T,
                )
            except Exception as e:
                print(f"{name} ({method}): cost evaluation failed: {e}")
                continue

            label = f"{name} ({method})"
            print(f"{label}:")
            print(f"  Training loss (final): {hist_m[-1]['loss']:.4f}")
            print(f"  True continuous cost:  {true_cost:.4f}")
            if isinstance(dts_final, np.ndarray):
                print(f"  dt range: [{np.min(dts_final):.5f}, {np.max(dts_final):.5f}]")
                print(f"  dt std:   {np.std(dts_final):.5f}")
            print()


# ============================================================================ #
# Baseline Sweep (cell 8)
# ============================================================================ #

def plot_baseline_sweep(results_dir, show=False):
    """Sweep uniform-timestep CLQR over n=80..1000 and plot results."""
    interval = 80

    fig, axs = plt.subplots(12, 2, figsize=(6.4, 12.8), constrained_layout=True)

    i = 0
    for n_test in range(interval, N + 1, interval):
        dt_test = T / n_test
        prob_test, s_test, u_test = create_pann_clqr(n_test, s0, A, B, Q, R, dt_test, u_max)

        start_time = time.time()
        prob_test.solve()
        solve_time = time.time() - start_time

        print(
            f"n = {n_test}\n"
            f"Optimal cost: {prob_test.objective.value:.4f}, "
            f"solve time meas: {solve_time:.4f} s, "
            f"solve time: {prob_test.solver_stats.solve_time:.4f} s\n"
        )

        times = np.arange(n_test) * dt_test
        s_vec = np.array([sv.value for sv in s_test[1:n_test + 1]])
        u_vec = np.array([uv.value for uv in u_test[:n_test]])

        axs[2 * (i // 2), i % 2].plot(times, s_vec, label=['x', 'y', 'z'])
        axs[2 * (i // 2), i % 2].set(
            xlabel='Time', ylabel='State', title=fr"$n={n_test}$",
        )

        axs[2 * (i // 2) + 1, i % 2].plot(times, u_vec)
        axs[2 * (i // 2) + 1, i % 2].set(
            xlabel='Time', ylabel='Input',
        )

        i = i + 1

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "baseline_sweep.pdf"), bbox_inches='tight')

    if not show:
        plt.close(fig)


# ============================================================================ #
# Cross-Method Analysis (cells 40, 42, 44)
# ============================================================================ #

def _build_method_solutions(all_results, n):
    """Build the method_solutions dict used for density/cross-correlation plots."""
    method_solutions = {}

    # Define the mapping from result keys to display config
    config = {
        "aux": ("Aux", 1),
        "rep": ("Rep", 2),
        "hs": ("HS", 2),
        "zoh": ("ZOH", 2),
    }

    for key, result in all_results.items():
        prefix, sol_method = config.get(key, (key, 2))
        n_method = result["n"]
        sol_dict = result["sol"]
        history = result["history"]

        for method in result["internal_methods"]:
            label = f"{prefix} (n={n_method}): {method}" if n_method != n else f"{prefix}: {method}"
            method_solutions[label] = {
                'sol': sol_dict[method],
                'history': [h for h in history if h['method'] == method],
                'sol_method': sol_method,
                'n': n_method,
            }

    return method_solutions


def plot_density_analysis(method_solutions, colors, results_dir, show=False):
    """Plot sampling density vs trajectory changes for all methods (cell 42)."""
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


# ============================================================================ #
# Timestep Evolution Videos
# ============================================================================ #

# Mapping from (result_key, internal_method) to (dts_dist_pickle, video_name)
_VIDEO_CONFIG = {
    ("aux", "time scaled"): ("dts_dist_aux_time_scaled", "aux_time_scaled_timesteps"),
    ("aux", "time bar scaled"): ("dts_dist_aux_time_bar_scaled", "aux_time_bar_scaled_timesteps"),
    ("rep", "time scaled"): ("dts_dist_rep_time_scaled", "rep_timesteps"),
    ("hs", "uniform_resample"): ("dts_dist_hs_uniform_resample", "hs_uniform_timesteps"),
    ("hs", "substeps"): ("dts_dist_hs_substeps", "hs_substeps_timesteps"),
}


def _save_all_videos(all_results, data_dir, results_dir, n, n_zoh, suffix=""):
    """Generate timestep evolution videos from saved dts distribution pickles."""
    for name, result in all_results.items():
        for method in result["internal_methods"]:
            # Look up config, with special handling for zoh
            if name == "zoh":
                dist_pkl = pickle_name("dts_dist_zoh" + suffix, n_zoh, n)
                video_name = f"zoh_timesteps_n{n_zoh}" if n_zoh != n else "zoh_timesteps"
            else:
                key = (name, method)
                if key not in _VIDEO_CONFIG:
                    continue
                dist_pkl, video_name = _VIDEO_CONFIG[key]
                dist_pkl = dist_pkl + suffix

            try:
                dts_all = load_pickle(data_dir, dist_pkl)
            except (FileNotFoundError, OSError):
                print(f"  No dts distribution for {name}/{method}, skipping video")
                continue

            print(f"  Video: {video_name}")
            save_timesteps_video(results_dir, video_name, dts_all=dts_all)


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Visualization for differentiable time optimization experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle source directory (default: data/pann_clqr_dt)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Plot output directory (default: results/pann_clqr_dt)",
    )
    parser.add_argument(
        "--method", nargs="+", default=None,
        help="Which methods to plot (default: all available)",
    )
    parser.add_argument("--analysis-only", action="store_true", help="Only cross-method analysis plots")
    parser.add_argument("--n", type=int, default=160, help="Timestep count for pickle naming")
    parser.add_argument("--n-zoh", type=int, default=80, help="ZOH timestep count")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--baseline", action="store_true", help="Include baseline sweep plot")
    parser.add_argument("--save-video", action="store_true", help="Save timestep evolution videos")
    parser.add_argument(
        "--reparam", default="both",
        choices=[*REPARAM_CHOICES, "both"],
        help="Which reparametrization results to plot (default: both)",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "pann_clqr_dt")
    results_dir_base = args.results_dir or os.path.join(script_dir, "results", "pann_clqr_dt")

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
        all_results = load_all_results(data_dir, n=args.n, n_zoh=args.n_zoh,
                                       suffix=suffix)

        if not all_results:
            print(f"No {reparam} results found. Skipping.")
            continue

        # Filter methods if requested
        if args.method:
            all_results = {k: v for k, v in all_results.items() if k in args.method}

        loaded = list(all_results.keys())
        print(f"Loaded methods: {loaded}")

        if not args.analysis_only:
            # Per-method plots
            for name, result in all_results.items():
                print(f"Plotting {name}...")
                plot_method_results(name, result, results_dir, show=args.show)

            # HS comparison
            if "hs" in all_results:
                print("Plotting HS comparison...")
                plot_hs_comparison(all_results["hs"], args.n, results_dir, show=args.show)

            # Continuous costs
            print_continuous_costs(all_results, args.n)

            # Baseline sweep
            if args.baseline:
                print("Running baseline sweep...")
                plot_baseline_sweep(results_dir, show=args.show)

        # Cross-method analysis
        method_solutions = _build_method_solutions(all_results, args.n)
        if method_solutions:
            print(f"Analysis: {len(method_solutions)} method variants")
            plot_density_analysis(method_solutions, colors, results_dir, show=args.show)

        # Timestep evolution videos
        if args.save_video:
            print("Saving timestep evolution videos...")
            _save_all_videos(all_results, data_dir, results_dir, args.n, args.n_zoh,
                             suffix=suffix)

        print(f"\nPlots saved to: {results_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
