#!/usr/bin/env python
"""Visualization script for inverted pendulum differentiable time optimization.

Loads pickles from data/invpend_dt/ and produces plots in results/invpend_dt/.

Usage:
    python plot_invpend_dt.py
    python plot_invpend_dt.py --method rep zoh
    python plot_invpend_dt.py --analysis-only
    python plot_invpend_dt.py --show
    python plot_invpend_dt.py --baseline
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from invpend_clqr import create_invpend_baseline_clqr
from utils import (
    LOSS_REGISTRY,
    compute_trajectory_metrics,
    evaluate_continuous_cost,
    extract_trajectory_data,
    load_pickle,
    plot_cross_correlations,
    plot_density_and_changes,
    plot_training_res,
    save_training_res,
)


# ============================================================================ #
# System Constants (Linearized Cart-Pole) — needed for baseline + cost eval
# ============================================================================ #

m_p, M_c, l_p, g_val = 0.2, 1.0, 0.5, 9.81

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m_p * g_val / M_c, 0],
    [0, 0, 0, 1],
    [0, 0, (M_c + m_p) * g_val / (M_c * l_p), 0],
])
B = np.array([[0], [1 / M_c], [0], [-1 / (M_c * l_p)]])

s0 = np.array([0.0, 0.0, 0.1, 0.0])
s_goal = np.array([5.0, 0.0, 0.0, 0.0])
T = 5.0
Q = np.diag([1.0, 0.1, 10.0, 0.1])
R = 0.01 * np.eye(1)
u_max = 10.0
v_max = 2.0
theta_max = 0.15
x_max = {1: v_max, 2: theta_max}
n_s = 4
n_u = 1
e0 = s0 - s_goal

STATE_LABELS = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']


# ============================================================================ #
# Method Configuration
# ============================================================================ #

@dataclass(frozen=True)
class MethodConfig:
    key: str
    sol_pickle: str
    history_pickle: str
    internal_methods: tuple[str, ...]


METHOD_CONFIGS: list[MethodConfig] = [
    MethodConfig("rep", "sol_rep", "history_rep", ("time_scaled",)),
    MethodConfig("zoh", "sol_zoh", "history_zoh", ("exact_zoh_integrated",)),
]


# ============================================================================ #
# Load Results
# ============================================================================ #

def load_method_results(data_dir):
    """Load all available method results from pickle files.

    Returns:
        results: dict of {method_name: {sol, history, n, internal_methods}}
    """
    results = {}

    for cfg in METHOD_CONFIGS:
        try:
            sol = load_pickle(data_dir, cfg.sol_pickle)
            history = load_pickle(data_dir, cfg.history_pickle)
            n_inferred = len(np.array(history[-1]['dts']).flatten())
            results[cfg.key] = {
                "sol": sol,
                "history": history,
                "n": n_inferred,
                "internal_methods": list(cfg.internal_methods),
            }
        except (FileNotFoundError, OSError):
            pass

    return results


def load_loss_results(data_dir, loss_names=None):
    """Load all available loss results from pickle files.

    Returns:
        results: dict of {loss_name: {"sol": ..., "history": ...}}
    """
    candidates = loss_names or list(LOSS_REGISTRY.keys())
    results = {}

    for loss_name in candidates:
        try:
            sol = load_pickle(data_dir, f"sol_{loss_name}")
            history = load_pickle(data_dir, f"history_{loss_name}")
            n_inferred = len(np.array(history[-1]['dts']).flatten())
            results[loss_name] = {"sol": sol, "history": history, "n": n_inferred}
        except (FileNotFoundError, OSError):
            pass

    return results


# ============================================================================ #
# Per-Method Plots
# ============================================================================ #

def plot_method_results(name, result, results_dir, show=False):
    """Plot training results for a single method (2x3 grid per sub-method)."""
    sol_dict = result["sol"]
    history = result["history"]
    n_method = result["n"]
    internal_methods = result["internal_methods"]

    for method in internal_methods:
        sol = sol_dict[method]
        hist_m = [h for h in history if h['method'] == method]
        if not hist_m:
            print(f"  No history for {name}/{method}, skipping")
            continue

        plot_training_res(sol, hist_m, n_method, sol_method=2)
        plt.suptitle(f"{name}: {method}")

        if results_dir:
            _key = method.replace(' ', '_')
            save_training_res(results_dir, f"{name}_{_key}", sol, hist_m,
                              n_method, sol_method=2)

        if not show:
            plt.close('all')


# ============================================================================ #
# Inverted Pendulum Trajectory Plot (NEW)
# ============================================================================ #

def plot_invpend_trajectory(sol, dts, n, title=None, results_dir=None,
                            filename=None, show=False):
    """Plot inverted pendulum trajectory with state constraints.

    4 panels: position, angle, velocity, force.
    Timestep boundaries overlaid as vertical lines.

    Args:
        sol: raw layer output (error coordinates)
        dts: final timestep array, shape (n,)
        n: number of timesteps
        title: optional figure title
        results_dir: directory to save figure
        filename: base filename for saving
        show: whether to keep figure open
    """
    e_arr = np.array([sol[i].detach().numpy() for i in range(n)])
    u_arr = np.array([sol[n + i].detach().numpy() for i in range(n)])
    times = np.concatenate([[0.0], np.cumsum(dts)])

    # Prepend initial state and convert to actual states
    s_arr = np.vstack([e0, e_arr]) + s_goal
    u_arr_plot = np.vstack([u_arr[:1], u_arr])  # repeat u_0 for t=0

    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.4), constrained_layout=True)

    def _add_timestep_lines(ax):
        for t_val in times:
            ax.axvline(t_val, color='gray', linestyle='-', alpha=0.08,
                       linewidth=0.5)

    # Position x(t)
    axs[0, 0].plot(times, s_arr[:, 0])
    axs[0, 0].axhline(s_goal[0], color='gray', linestyle='--', alpha=0.5,
                       label=f'goal={s_goal[0]}')
    axs[0, 0].set(xlabel='Time [s]', ylabel='x [m]', title='Position')
    axs[0, 0].legend(fontsize=7)
    _add_timestep_lines(axs[0, 0])

    # Angle theta(t)
    axs[0, 1].plot(times, s_arr[:, 2])
    axs[0, 1].axhline(theta_max, color='r', linestyle='--', alpha=0.5,
                       label=r'$\pm\theta_{\max}$')
    axs[0, 1].axhline(-theta_max, color='r', linestyle='--', alpha=0.5)
    axs[0, 1].set(xlabel='Time [s]', ylabel=r'$\theta$ [rad]', title='Angle')
    axs[0, 1].legend(fontsize=7)
    _add_timestep_lines(axs[0, 1])

    # Velocity dx(t)
    axs[1, 0].plot(times, s_arr[:, 1])
    axs[1, 0].axhline(v_max, color='r', linestyle='--', alpha=0.5,
                       label=r'$\pm v_{\max}$')
    axs[1, 0].axhline(-v_max, color='r', linestyle='--', alpha=0.5)
    axs[1, 0].set(xlabel='Time [s]', ylabel=r'$\dot{x}$ [m/s]',
                   title='Velocity')
    axs[1, 0].legend(fontsize=7)
    _add_timestep_lines(axs[1, 0])

    # Force F(t)
    axs[1, 1].plot(times, u_arr_plot.flatten())
    axs[1, 1].axhline(u_max, color='r', linestyle='--', alpha=0.5,
                       label=r'$\pm F_{\max}$')
    axs[1, 1].axhline(-u_max, color='r', linestyle='--', alpha=0.5)
    axs[1, 1].set(xlabel='Time [s]', ylabel='F [N]', title='Force')
    axs[1, 1].legend(fontsize=7)
    _add_timestep_lines(axs[1, 1])

    # Timestep distribution (plot at interval end-times)
    axs[0, 2].plot(times[1:], dts)
    axs[0, 2].axhline(T / n, color='gray', linestyle='--', alpha=0.5,
                       label='uniform')
    axs[0, 2].set(xlabel='Time [s]', ylabel=r'$\Delta t$',
                   title='Timestep Distribution')
    axs[0, 2].legend(fontsize=7)

    # Angular velocity dtheta(t)
    axs[1, 2].plot(times, s_arr[:, 3])
    axs[1, 2].set(xlabel='Time [s]', ylabel=r'$\dot{\theta}$ [rad/s]',
                   title='Angular Velocity')
    _add_timestep_lines(axs[1, 2])

    if title:
        fig.suptitle(title, fontsize=13)

    if results_dir and filename:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, f"{filename}.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)

    return fig


# ============================================================================ #
# Continuous Cost Evaluation
# ============================================================================ #

def print_continuous_costs(method_results, loss_results):
    """Evaluate and print true continuous-time costs."""
    print("=== True Continuous-Time Cost Comparison ===\n")

    # Methods
    for name, result in method_results.items():
        sol_dict = result["sol"]
        history = result["history"]
        n_method = result["n"]

        for method in result["internal_methods"]:
            sol = sol_dict[method]
            hist_m = [h for h in history if h['method'] == method]
            if not hist_m:
                continue

            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_method + i].detach().float()
                         for i in range(n_method)]

            try:
                true_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, e0, A, B, Q, R, T,
                )
            except Exception as exc:
                print(f"{name} ({method}): cost evaluation failed: {exc}")
                continue

            print(f"{name} ({method}):")
            print(f"  Training loss (final): {hist_m[-1]['loss']:.4f}")
            print(f"  True continuous cost:  {true_cost:.4f}")
            if isinstance(dts_final, np.ndarray):
                print(f"  dt range: [{np.min(dts_final):.5f}, "
                      f"{np.max(dts_final):.5f}]")
                print(f"  dt std:   {np.std(dts_final):.5f}")
            print()

    # Alt losses
    for loss_name, result in loss_results.items():
        sol = result["sol"]
        history = result["history"]
        n_loss = result["n"]

        dts_final = history[-1]['dts']
        inputs_qp = [sol[n_loss + k].detach().float() for k in range(n_loss)]

        try:
            true_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, e0, A, B, Q, R, T,
            )
        except Exception as exc:
            print(f"{loss_name}: cost evaluation failed: {exc}")
            continue

        print(f"{loss_name}:")
        print(f"  Training loss (final):  {history[-1]['loss']:.4f}")
        print(f"  Loss OCP (final):       {history[-1]['loss_ocp']:.4f}")
        print(f"  Loss reg (final):       {history[-1]['loss_reg']:.4f}")
        print(f"  Lambda hat (final):     {history[-1]['lambda_hat']:.4f}")
        print(f"  True continuous cost:   {true_cost:.4f}")
        if isinstance(dts_final, np.ndarray):
            print(f"  dt range: [{np.min(dts_final):.5f}, "
                  f"{np.max(dts_final):.5f}]")
            print(f"  dt std:   {np.std(dts_final):.5f}")
        print()


# ============================================================================ #
# Baseline Sweep
# ============================================================================ #

def plot_baseline_sweep(results_dir, show=False, n_tests=None):
    """Sweep uniform-timestep CLQR over n=40..200 and plot results."""
    if n_tests is None:
        n_tests = list(range(40, 201, 20))
    n_rows = len(n_tests)

    fig, axs = plt.subplots(n_rows, 2, figsize=(9.6, 2.4 * n_rows),
                            constrained_layout=True)

    for i, n_test in enumerate(n_tests):
        dt_test = T / n_test
        try:
            prob, s_test, u_test = create_invpend_baseline_clqr(
                n_test, s0, A, B, Q, R, dt_test, u_max, x_max, s_goal,
            )
            start_time = time.time()
            prob.solve()
            solve_time = time.time() - start_time

            if prob.status not in ("optimal", "optimal_inaccurate"):
                print(f"n={n_test}: {prob.status}")
                axs[i, 0].text(0.5, 0.5, f"n={n_test}: {prob.status}",
                               transform=axs[i, 0].transAxes, ha='center')
                axs[i, 1].text(0.5, 0.5, "infeasible",
                               transform=axs[i, 1].transAxes, ha='center')
                continue

            cost = prob.objective.value
            print(f"n={n_test}: cost={cost:.4f}, "
                  f"solve={solve_time:.4f}s")

            times = np.arange(n_test) * dt_test
            e_vec = np.array([sv.value for sv in s_test[1:n_test + 1]])
            s_vec = e_vec + s_goal
            u_vec = np.array([uv.value for uv in u_test[:n_test]])

            axs[i, 0].plot(times, s_vec, label=STATE_LABELS)
            axs[i, 0].set(xlabel='Time', ylabel='State',
                          title=f"n={n_test}, cost={cost:.4f}")
            if i == 0:
                axs[i, 0].legend(fontsize=6)

            axs[i, 1].plot(times, u_vec)
            axs[i, 1].set(xlabel='Time', ylabel='Force [N]')

        except Exception as exc:
            print(f"n={n_test}: failed ({exc})")
            axs[i, 0].text(0.5, 0.5, f"n={n_test}: error",
                           transform=axs[i, 0].transAxes, ha='center')

    fig.suptitle("Baseline Sweep: Uniform-dt CLQR", fontsize=13)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "baseline_sweep.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


# ============================================================================ #
# Cross-Method/Loss Analysis
# ============================================================================ #

def _build_method_solutions(method_results, loss_results):
    """Build the method_solutions dict for density/cross-correlation plots."""
    method_solutions = {}

    for key, result in method_results.items():
        sol_dict = result["sol"]
        history = result["history"]
        n_method = result["n"]

        for method in result["internal_methods"]:
            label = f"{key.upper()}: {method}"
            method_solutions[label] = {
                'sol': sol_dict[method],
                'history': [h for h in history if h['method'] == method],
                'sol_method': 2,
                'n': n_method,
            }

    for loss_name, result in loss_results.items():
        method_solutions[loss_name] = {
            'sol': result['sol'],
            'history': result['history'],
            'sol_method': 2,
            'n': result['n'],
        }

    return method_solutions


def plot_density_analysis(method_solutions, colors, results_dir, show=False):
    """Plot sampling density vs trajectory changes."""
    n_methods = len(method_solutions)
    if n_methods == 0:
        return

    n_rows = int(np.ceil(n_methods / 2))
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.5 * n_rows),
                            squeeze=False)

    for i, (key, ms) in enumerate(method_solutions.items()):
        n_m = ms['n']
        data = extract_trajectory_data(ms, n_m)
        metrics = compute_trajectory_metrics(data, n_m, T)
        plot_density_and_changes(data, metrics, key, colors,
                                axes=axs[i // 2, i % 2])

    for j in range(i + 1, n_rows * 2):
        fig.delaxes(axs[j // 2, j % 2])

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "density_analysis.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def plot_cross_correlation_analysis(method_solutions, colors, results_dir,
                                    show=False):
    """Plot cross-correlation for all methods/losses."""
    for key, ms in method_solutions.items():
        n_m = ms['n']
        data = extract_trajectory_data(ms, n_m)
        metrics = compute_trajectory_metrics(data, n_m, T)
        fig = plot_cross_correlations(data, metrics, key, colors, max_lag=30)

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            safe_key = (key.replace(" ", "_").replace(":", "")
                        .replace("(", "").replace(")", ""))
            fig.savefig(
                os.path.join(results_dir, f"cross_corr_{safe_key}.pdf"),
                bbox_inches='tight',
            )

        if not show:
            plt.close(fig)


# ============================================================================ #
# Loss Comparison
# ============================================================================ #

def plot_loss_results(loss_name, result, results_dir, show=False):
    """Plot training results for a single loss (2x3 grid)."""
    sol = result["sol"]
    history = result["history"]
    n = result["n"]

    plot_training_res(sol, history, n, sol_method=2)
    plt.suptitle(loss_name)

    if results_dir:
        save_training_res(results_dir, loss_name, sol, history, n, sol_method=2)

    if not show:
        plt.close('all')


def plot_loss_comparison(loss_results, results_dir, show=False):
    """Side-by-side timestep distributions for all losses."""
    if len(loss_results) <= 1:
        return

    n_losses = len(loss_results)
    fig, axes = plt.subplots(1, n_losses, figsize=(3.2 * n_losses, 3.2))
    if n_losses == 1:
        axes = [axes]

    for ax, (loss_name, result) in zip(axes, loss_results.items()):
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
        fig.savefig(os.path.join(results_dir, "loss_comparison.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def save_summary(method_results, loss_results, results_dir):
    """Compute and save metrics summary JSON."""
    summary = {}

    for name, result in method_results.items():
        for method in result["internal_methods"]:
            sol = result["sol"][method]
            hist_m = [h for h in result["history"]
                      if h['method'] == method]
            if not hist_m:
                continue
            n_m = result["n"]
            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_m + i].detach().float() for i in range(n_m)]
            try:
                cont_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, e0, A, B, Q, R, T)
            except Exception:
                cont_cost = None
            summary[f"{name}_{method}"] = {
                "continuous_cost": cont_cost,
                "final_loss": hist_m[-1]['loss'],
            }

    for loss_name, result in loss_results.items():
        sol = result["sol"]
        history = result["history"]
        n_loss = result["n"]
        dts_final = history[-1]['dts']
        inputs_qp = [sol[n_loss + k].detach().float() for k in range(n_loss)]
        try:
            cont_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, e0, A, B, Q, R, T)
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

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, metrics in summary.items():
        cost_str = (f"{metrics['continuous_cost']:.6f}"
                    if metrics['continuous_cost'] is not None else "N/A")
        print(f"\n{label}:")
        print(f"  Continuous cost:  {cost_str}")
        print(f"  Final loss:       {metrics['final_loss']:.6f}")

    return summary


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Visualization for inverted pendulum time optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle source directory (default: data/invpend_dt)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Plot output directory (default: results/invpend_dt)",
    )
    parser.add_argument(
        "--method", nargs="+", default=None,
        help="Which methods to plot (default: all available)",
    )
    parser.add_argument(
        "--loss", nargs="+", default=None,
        help="Which losses to plot (default: all available)",
    )
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only cross-method analysis plots")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--baseline", action="store_true",
                        help="Include baseline sweep plot")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "invpend_dt")
    results_dir = (args.results_dir
                   or os.path.join(script_dir, "results", "invpend_dt"))
    os.makedirs(results_dir, exist_ok=True)

    try:
        from dimp.utils import init_matplotlib, get_colors
        init_matplotlib()
        colors = get_colors()
    except ImportError:
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442',
                  '#D55E00']

    print(f"Loading data from: {data_dir}")
    method_results = load_method_results(data_dir)
    loss_results = load_loss_results(data_dir, loss_names=args.loss)

    # Filter methods if requested
    if args.method:
        method_results = {k: v for k, v in method_results.items()
                          if k in args.method}

    if not method_results and not loss_results:
        print("No results found. Run invpend_dt.py first to generate data.")
        return

    loaded_methods = list(method_results.keys())
    loaded_losses = list(loss_results.keys())
    print(f"Loaded methods: {loaded_methods}")
    print(f"Loaded losses: {loaded_losses}")

    if not args.analysis_only:
        # Per-method plots
        for name, result in method_results.items():
            print(f"Plotting {name}...")
            plot_method_results(name, result, results_dir,
                                show=args.show)

            # Inverted-pendulum-specific trajectory plot
            for method in result["internal_methods"]:
                sol = result["sol"][method]
                hist_m = [h for h in result["history"]
                          if h['method'] == method]
                if hist_m:
                    dts_final = np.array(hist_m[-1]['dts']).flatten()
                    plot_invpend_trajectory(
                        sol, dts_final, result["n"],
                        title=f"{name}: {method}",
                        results_dir=results_dir,
                        filename=f"trajectory_{name}_{method}",
                        show=args.show,
                    )

        # Per-loss plots
        for loss_name, result in loss_results.items():
            print(f"Plotting {loss_name}...")
            plot_loss_results(loss_name, result, results_dir,
                              show=args.show)

            # Inverted-pendulum trajectory for each loss
            dts_final = np.array(result["history"][-1]['dts']).flatten()
            plot_invpend_trajectory(
                result["sol"], dts_final, result["n"],
                title=loss_name,
                results_dir=results_dir,
                filename=f"trajectory_{loss_name}",
                show=args.show,
            )

        # Continuous costs
        print_continuous_costs(method_results, loss_results)

        # Loss comparison
        plot_loss_comparison(loss_results, results_dir, show=args.show)

        # Baseline sweep
        if args.baseline:
            print("Running baseline sweep...")
            plot_baseline_sweep(results_dir, show=args.show)

    # Cross-method/loss analysis
    method_solutions = _build_method_solutions(method_results, loss_results)
    if method_solutions:
        print(f"Analysis: {len(method_solutions)} variants")
        plot_density_analysis(method_solutions, colors, results_dir,
                              show=args.show)
        plot_cross_correlation_analysis(method_solutions, colors, results_dir,
                                        show=args.show)

    # Summary
    save_summary(method_results, loss_results, results_dir)

    print(f"\nPlots saved to: {results_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
