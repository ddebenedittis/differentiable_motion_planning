#!/usr/bin/env python
"""Visualization script for inverted pendulum differentiable time optimization.

Loads pickles from data/invpend_dt/ and produces plots in results/invpend_dt/.

Usage:
    python invpend_plot.py
    python invpend_plot.py --method rep zoh
    python invpend_plot.py --analysis-only
    python invpend_plot.py --show
    python invpend_plot.py --baseline
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from invpend_prob import (
    create_invpend_baseline_clqr,
    A, B, s0, s_goal, T, Q, R, u_max, v_max, theta_max, x_max, n_s, n_u, e0,
)
from utils import (
    LOSS_REGISTRY,
    MethodConfig,
    load_method_results,
    load_loss_results,
    plot_method_results,
    plot_loss_results,
    plot_loss_comparison,
    plot_density_analysis_grid,
    print_continuous_costs,
    save_summary,
    evaluate_continuous_cost,
    load_pickle,
    plot_training_res,
    save_training_res,
    extract_trajectory_data,
    compute_trajectory_metrics,
    plot_density_and_changes,
)


STATE_LABELS = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']


# ============================================================================ #
# Method Configuration
# ============================================================================ #

METHOD_CONFIGS: list[MethodConfig] = [
    MethodConfig("rep", "sol_rep", "history_rep", ("time_scaled",)),
    MethodConfig("zoh", "sol_zoh", "history_zoh", ("exact_zoh_integrated",)),
]


# ============================================================================ #
# Inverted Pendulum Trajectory Plot (NEW)
# ============================================================================ #

def plot_invpend_trajectory(dts, n, *, sol=None, s_arr=None, u_arr=None,
                            title=None, results_dir=None, filename=None,
                            show=False):
    """Plot inverted pendulum trajectory with state constraints.

    2x3 panels: position, angle, velocity, force, timestep distribution,
    angular velocity. Timestep boundaries overlaid as vertical lines.

    Provide either `sol` (raw layer output in error coordinates) or
    pre-computed `s_arr`/`u_arr` numpy arrays (for baseline plots).

    Args:
        dts: final timestep array, shape (n,)
        n: number of timesteps
        sol: raw layer output (error coordinates), optional
        s_arr: (n+1, n_s) actual states array, optional
        u_arr: (n, n_u) inputs array, optional
        title: optional figure title
        results_dir: directory to save figure
        filename: base filename for saving
        show: whether to keep figure open
    """
    if sol is not None:
        e_arr = np.array([sol[i].detach().numpy() for i in range(n)])
        u_arr = np.array([sol[n + i].detach().numpy() for i in range(n)])
        s_arr = np.vstack([e0, e_arr]) + s_goal
    u_arr_plot = np.vstack([u_arr[:1], u_arr])  # repeat u_0 for t=0
    times = np.concatenate([[0.0], np.cumsum(dts)])

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

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "invpend_dt")
    results_dir = (args.results_dir
                   or os.path.join(script_dir, "results", "invpend_dt"))

    try:
        from dimp.utils import init_matplotlib, get_colors
        init_matplotlib()
        colors = get_colors()
    except ImportError:
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442',
                  '#D55E00']

    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    method_results = load_method_results(data_dir, METHOD_CONFIGS)
    loss_results = load_loss_results(data_dir, loss_names=args.loss)

    # Filter methods if requested
    if args.method:
        method_results = {k: v for k, v in method_results.items()
                          if k in args.method}

    if not method_results and not loss_results:
        print("No results found.")
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
                        dts_final, result["n"], sol=sol,
                        title=f"{name}: {method}",
                        results_dir=results_dir,
                        filename=f"trajectory_{name}_{method}",
                        show=args.show,
                    )

        # Uniform-timestep baseline
        if method_results:
            n_uniform = next(iter(method_results.values()))["n"]
        elif loss_results:
            n_uniform = next(iter(loss_results.values()))["n"]
        else:
            n_uniform = 80
        dt_uniform = T / n_uniform
        try:
            prob, s_base, u_base = create_invpend_baseline_clqr(
                n_uniform, s0, A, B, Q, R, dt_uniform, u_max, x_max, s_goal,
            )
            prob.solve()
            if prob.status in ("optimal", "optimal_inaccurate"):
                print(f"--- Uniform: OCP cost ---")
                print(f"  n={n_uniform}, cost={prob.objective.value:.6f}")
                dts_uniform = np.full(n_uniform, dt_uniform)
                e_arr = np.array([s_base[i + 1].value.flatten()
                                  for i in range(n_uniform)])
                s_base_arr = np.vstack([s0, e_arr + s_goal])
                u_base_arr = np.array([u_base[i].value.flatten()
                                       for i in range(n_uniform)])
                plot_invpend_trajectory(
                    dts_uniform, n_uniform,
                    s_arr=s_base_arr, u_arr=u_base_arr,
                    title=(f"Uniform baseline (n={n_uniform}, "
                           f"cost={prob.objective.value:.4f})"),
                    results_dir=results_dir,
                    filename="trajectory_uniform_baseline",
                    show=args.show,
                )
            else:
                print(f"Uniform baseline: {prob.status}")
        except Exception as exc:
            print(f"Uniform baseline failed: {exc}")

        # Per-loss plots
        for loss_name, result in loss_results.items():
            print(f"Plotting {loss_name}...")
            print(f"  OCP cost: {result['history'][-1]['loss_ocp']:.6f}")
            plot_loss_results(loss_name, result, results_dir,
                              show=args.show)

            # Inverted-pendulum trajectory for each loss
            dts_final = np.array(result["history"][-1]['dts']).flatten()
            plot_invpend_trajectory(
                dts_final, result["n"], sol=result["sol"],
                title=loss_name,
                results_dir=results_dir,
                filename=f"trajectory_{loss_name}",
                show=args.show,
            )

        # Continuous costs
        print_continuous_costs(method_results, loss_results,
                              s0_eval=e0, A=A, B=B, Q=Q, R=R, T=T)

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
        plot_density_analysis_grid(method_solutions, T, colors, results_dir,
                                  show=args.show)

    # Summary
    save_summary(method_results, loss_results, results_dir,
                 s0_eval=e0, A=A, B=B, Q=Q, R=R, T=T)

    print(f"\nPlots saved to: {results_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
