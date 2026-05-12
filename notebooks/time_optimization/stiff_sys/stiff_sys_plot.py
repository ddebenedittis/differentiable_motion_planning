#!/usr/bin/env python
"""Visualization for stiff system LTI differentiable time optimization.

Loads pickles from data/stiff_sys_dt/ and produces plots in results/stiff_sys_dt/.

Usage:
    python stiff_sys_plot.py
    python stiff_sys_plot.py --method rep zoh
    python stiff_sys_plot.py --analysis-only
    python stiff_sys_plot.py --show
    python stiff_sys_plot.py --baseline
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np

from stiff_sys_prob import (
    create_stiff_sys_baseline_clqr,
    A, B, s0, T, Q, R, u_max, x_max, n_s, n_u,
)
from stiff_sys_train import SPEC
from utils import MethodConfig, run_plot_main


STATE_LABELS = [
    r'$x_1$ ($\tau$=0.1s)',
    r'$x_2$ ($\tau$=10s)',
    r'$x_3$ ($\tau$=100s)',
]


METHOD_CONFIGS = [
    MethodConfig("rep", "sol_rep", "history_rep", ("time_scaled",)),
    MethodConfig("zoh", "sol_zoh", "history_zoh", ("exact_zoh_integrated",)),
]


# ============================================================================ #
# Stiff System Trajectory Plot
# ============================================================================ #

def _add_zoom_inset(parent_ax, fig, x_data, y_data, color, *,
                    zoom_xlim, loc, time_lines=None):
    inset_bg = Rectangle(
        (loc[0], loc[1]), loc[2], loc[3],
        transform=parent_ax.transAxes,
        facecolor='white', edgecolor='none', alpha=0.85, zorder=15,
    )
    parent_ax.add_patch(inset_bg)

    axins = parent_ax.inset_axes(loc)
    axins.set_zorder(20)
    axins.set_facecolor('none')
    axins.plot(x_data, y_data, color=color)

    if time_lines is not None:
        for t_val in time_lines:
            if zoom_xlim[0] <= t_val <= zoom_xlim[1]:
                axins.axvline(t_val, color='gray', linestyle='-',
                              alpha=0.08, linewidth=0.5)

    axins.set_xlim(zoom_xlim)

    x_arr = np.asarray(x_data)
    y_arr = np.asarray(y_data)
    mask = (x_arr >= zoom_xlim[0]) & (x_arr <= zoom_xlim[1])
    if mask.any():
        idx = np.where(mask)[0]
        lo = max(0, idx.min() - 1)
        hi = min(len(x_arr), idx.max() + 2)
        y_window = y_arr[lo:hi]
        y_lo, y_hi = float(np.min(y_window)), float(np.max(y_window))
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.5 * max(abs(y_lo), 1.0)
        axins.set_ylim(y_lo - pad, y_hi + pad)

    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    axins.tick_params(labelsize=8)

    main_ylim = parent_ax.get_ylim()
    source_rect = Rectangle(
        (zoom_xlim[0], main_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        main_ylim[1] - main_ylim[0],
        edgecolor='black', facecolor='none',
        linewidth=1.2, linestyle='--', zorder=5, clip_on=False,
    )
    parent_ax.add_patch(source_rect)

    con_top = ConnectionPatch(
        xyA=(zoom_xlim[1], main_ylim[1]), xyB=(0, 1),
        coordsA='data', coordsB='axes fraction',
        axesA=parent_ax, axesB=axins,
        color='black', linewidth=1.0, linestyle='--', zorder=21,
    )
    con_bot = ConnectionPatch(
        xyA=(zoom_xlim[1], main_ylim[0]), xyB=(0, 0),
        coordsA='data', coordsB='axes fraction',
        axesA=parent_ax, axesB=axins,
        color='black', linewidth=1.0, linestyle='--', zorder=21,
    )
    fig.add_artist(con_top)
    fig.add_artist(con_bot)


def plot_stiff_sys_trajectory(dts, n, *, sol=None, s_arr=None, u_arr=None,
                              title=None, results_dir=None, filename=None,
                              show=False):
    """Multi-scale state/input trajectory plot with zoom insets on fast modes."""
    if sol is not None:
        s_arr_raw = np.array([sol[i].detach().numpy() for i in range(n)])
        u_arr = np.array([sol[n + i].detach().numpy() for i in range(n)])
        s_arr = np.vstack([s0, s_arr_raw])
    u_arr_plot = np.vstack([u_arr[:1], u_arr])
    times = np.concatenate([[0.0], np.cumsum(dts)])

    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.4), constrained_layout=True)

    def _add_timestep_lines(ax):
        for t_val in times:
            ax.axvline(t_val, color='gray', linestyle='-', alpha=0.08,
                       linewidth=0.5)

    line_x1, = axs[0, 0].plot(times, s_arr[:, 0])
    axs[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[0, 0].set(xlabel='Time [s]', ylabel=r'$x_1$',
                  title=r'$x_1$ (fast, $\tau$=0.1s)')
    _add_timestep_lines(axs[0, 0])
    _add_zoom_inset(axs[0, 0], fig, times, s_arr[:, 0], line_x1.get_color(),
                    zoom_xlim=(0.0, 0.5), loc=[0.40, 0.08, 0.55, 0.55],
                    time_lines=times)

    axs[0, 1].plot(times, s_arr[:, 1])
    axs[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[0, 1].set(xlabel='Time [s]', ylabel=r'$x_2$',
                  title=r'$x_2$ (medium, $\tau$=10s)')
    _add_timestep_lines(axs[0, 1])

    axs[0, 2].plot(times, s_arr[:, 2])
    axs[0, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[0, 2].set(xlabel='Time [s]', ylabel=r'$x_3$',
                  title=r'$x_3$ (slow, $\tau$=100s)')
    _add_timestep_lines(axs[0, 2])

    line_u, = axs[1, 0].plot(times, u_arr_plot.flatten())
    axs[1, 0].axhline(u_max, color='r', linestyle='--', alpha=0.5,
                      label=r'$\pm u_{\max}$')
    axs[1, 0].axhline(-u_max, color='r', linestyle='--', alpha=0.5)
    axs[1, 0].set(xlabel='Time [s]', ylabel='u', title='Input')
    axs[1, 0].legend(fontsize=7)
    _add_timestep_lines(axs[1, 0])
    _add_zoom_inset(axs[1, 0], fig, times, u_arr_plot.flatten(),
                    line_u.get_color(),
                    zoom_xlim=(0.0, 0.5), loc=[0.40, 0.40, 0.55, 0.55],
                    time_lines=times)

    axs[1, 1].plot(times[1:], dts)
    axs[1, 1].axhline(T / n, color='gray', linestyle='--', alpha=0.5,
                      label='uniform')
    axs[1, 1].set(xlabel='Time [s]', ylabel=r'$\Delta t$',
                  title='Timestep Distribution')
    axs[1, 1].legend(fontsize=7)

    axs[1, 2].hist(dts.flatten(), bins=30, alpha=0.7, edgecolor='black')
    axs[1, 2].axvline(T / n, color='r', linestyle='--',
                      label=f'uniform={T/n:.4f}')
    axs[1, 2].set(xlabel=r'$\Delta t$', ylabel='Count',
                  title='Timestep Histogram')
    axs[1, 2].legend(fontsize=7)

    if title:
        fig.suptitle(title, fontsize=13)

    if results_dir and filename:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, f"{filename}.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def plot_baseline_sweep(results_dir, show=False, n_tests=None):
    if n_tests is None:
        n_tests = list(range(20, 201, 20))
    n_rows = len(n_tests)

    fig, axs = plt.subplots(n_rows, 2, figsize=(9.6, 2.4 * n_rows),
                            constrained_layout=True)

    for i, n_test in enumerate(n_tests):
        dt_test = T / n_test
        try:
            prob, s_test, u_test = create_stiff_sys_baseline_clqr(
                n_test, s0, A, B, Q, R, dt_test, u_max, x_max,
            )
            t0 = time.time()
            prob.solve()
            solve_time = time.time() - t0

            if prob.status not in ("optimal", "optimal_inaccurate"):
                print(f"n={n_test}: {prob.status}")
                continue

            cost = prob.objective.value
            print(f"n={n_test}: cost={cost:.4f}, solve={solve_time:.4f}s")
            times = np.arange(n_test) * dt_test
            s_vec = np.array([sv.value for sv in s_test[1:n_test + 1]])
            u_vec = np.array([uv.value for uv in u_test[:n_test]])

            axs[i, 0].plot(times, s_vec, label=STATE_LABELS)
            axs[i, 0].set(xlabel='Time', ylabel='State',
                          title=f"n={n_test}, cost={cost:.4f}")
            if i == 0:
                axs[i, 0].legend(fontsize=6)
            axs[i, 1].plot(times, u_vec)
            axs[i, 1].set(xlabel='Time', ylabel='Input')
        except Exception as exc:
            print(f"n={n_test}: failed ({exc})")

    fig.suptitle("Baseline Sweep: Uniform-dt CLQR", fontsize=13)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "baseline_sweep.pdf"),
                    bbox_inches='tight')
    if not show:
        plt.close(fig)


def _baseline_factory(n):
    return create_stiff_sys_baseline_clqr(n, s0, A, B, Q, R, T / n,
                                          u_max, x_max)


def _baseline_sweep_step(args, results_dir, method_results, loss_results):
    if args.baseline:
        print("Running baseline sweep...")
        plot_baseline_sweep(results_dir, show=args.show)


if __name__ == "__main__":
    run_plot_main(
        SPEC, METHOD_CONFIGS,
        trajectory_plot_fn=plot_stiff_sys_trajectory,
        baseline_factory=_baseline_factory,
        baseline_n_fallback=40,
        extra_steps=[_baseline_sweep_step],
    )
