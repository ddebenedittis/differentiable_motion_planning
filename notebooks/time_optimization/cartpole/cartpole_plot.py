#!/usr/bin/env python
"""Visualization for cart-pole differentiable time optimization.

Loads pickles from data/cartpole_dt/ and produces plots in results/cartpole/.

Usage:
    python cartpole_plot.py
    python cartpole_plot.py --method rep zoh
    python cartpole_plot.py --analysis-only
    python cartpole_plot.py --show
    python cartpole_plot.py --baseline
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from cartpole_prob import (
    create_cartpole_baseline_clqr,
    A, B, s0, s_goal, T, Q, R, u_max, v_max, theta_max, x_max, n_s, n_u, e0,
)
from cartpole_train import SPEC
from utils import MethodConfig, run_plot_main


STATE_LABELS = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']


METHOD_CONFIGS = [
    MethodConfig("rep", "sol_rep", "history_rep", ("time_scaled",)),
    MethodConfig("zoh", "sol_zoh", "history_zoh", ("exact_zoh_integrated",)),
]


# ============================================================================ #
# Cart-Pole Trajectory Plot
# ============================================================================ #

def plot_cartpole_trajectory(dts, n, *, sol=None, s_arr=None, u_arr=None,
                             title=None, results_dir=None, filename=None,
                             show=False):
    """Cart-pole trajectory plot with state-constraint overlays."""
    if sol is not None:
        e_arr = np.array([sol[i].detach().numpy() for i in range(n)])
        u_arr = np.array([sol[n + i].detach().numpy() for i in range(n)])
        s_arr = np.vstack([e0, e_arr]) + s_goal
    u_arr_plot = np.vstack([u_arr[:1], u_arr])
    times = np.concatenate([[0.0], np.cumsum(dts)])

    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.4), constrained_layout=True)

    def _add_timestep_lines(ax):
        for t_val in times:
            ax.axvline(t_val, color='gray', linestyle='-', alpha=0.08,
                       linewidth=0.5)

    axs[0, 0].plot(times, s_arr[:, 0])
    axs[0, 0].axhline(s_goal[0], color='gray', linestyle='--', alpha=0.5,
                      label=f'goal={s_goal[0]}')
    axs[0, 0].set(xlabel='Time [s]', ylabel='x [m]', title='Position')
    axs[0, 0].legend(fontsize=7)
    _add_timestep_lines(axs[0, 0])

    axs[0, 1].plot(times, s_arr[:, 2])
    axs[0, 1].axhline(theta_max, color='r', linestyle='--', alpha=0.5,
                      label=r'$\pm\theta_{\max}$')
    axs[0, 1].axhline(-theta_max, color='r', linestyle='--', alpha=0.5)
    axs[0, 1].set(xlabel='Time [s]', ylabel=r'$\theta$ [rad]', title='Angle')
    axs[0, 1].legend(fontsize=7)
    _add_timestep_lines(axs[0, 1])

    axs[1, 0].plot(times, s_arr[:, 1])
    axs[1, 0].axhline(v_max, color='r', linestyle='--', alpha=0.5,
                      label=r'$\pm v_{\max}$')
    axs[1, 0].axhline(-v_max, color='r', linestyle='--', alpha=0.5)
    axs[1, 0].set(xlabel='Time [s]', ylabel=r'$\dot{x}$ [m/s]',
                  title='Velocity')
    axs[1, 0].legend(fontsize=7)
    _add_timestep_lines(axs[1, 0])

    axs[1, 1].plot(times, u_arr_plot.flatten())
    axs[1, 1].axhline(u_max, color='r', linestyle='--', alpha=0.5,
                      label=r'$\pm F_{\max}$')
    axs[1, 1].axhline(-u_max, color='r', linestyle='--', alpha=0.5)
    axs[1, 1].set(xlabel='Time [s]', ylabel='F [N]', title='Force')
    axs[1, 1].legend(fontsize=7)
    _add_timestep_lines(axs[1, 1])

    axs[0, 2].plot(times[1:], dts)
    axs[0, 2].axhline(T / n, color='gray', linestyle='--', alpha=0.5,
                      label='uniform')
    axs[0, 2].set(xlabel='Time [s]', ylabel=r'$\Delta t$',
                  title='Timestep Distribution')
    axs[0, 2].legend(fontsize=7)

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


def plot_baseline_sweep(results_dir, show=False, n_tests=None):
    if n_tests is None:
        n_tests = list(range(40, 201, 20))
    n_rows = len(n_tests)

    fig, axs = plt.subplots(n_rows, 2, figsize=(9.6, 2.4 * n_rows),
                            constrained_layout=True)

    for i, n_test in enumerate(n_tests):
        dt_test = T / n_test
        try:
            prob, s_test, u_test = create_cartpole_baseline_clqr(
                n_test, s0, A, B, Q, R, dt_test, u_max, x_max, s_goal,
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

    fig.suptitle("Baseline Sweep: Uniform-dt CLQR", fontsize=13)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "baseline_sweep.pdf"),
                    bbox_inches='tight')
    if not show:
        plt.close(fig)


class _CartpoleBaselineProb:
    """Wraps cart-pole baseline so the runner sees actual states (not error)."""
    __slots__ = ("inner", "_s_actual", "_u")

    def __init__(self, n):
        prob, s_test, u_test = create_cartpole_baseline_clqr(
            n, s0, A, B, Q, R, T / n, u_max, x_max, s_goal,
        )
        self.inner = prob
        # Wrap state vars in proxies that return actual state values when solved.
        self._s_actual = [_StateActualProxy(sv) for sv in s_test]
        self._u = u_test

    @property
    def status(self):
        return self.inner.status

    @property
    def objective(self):
        return self.inner.objective

    def solve(self, *args, **kwargs):
        return self.inner.solve(*args, **kwargs)


class _StateActualProxy:
    __slots__ = ("_var",)

    def __init__(self, var):
        self._var = var

    @property
    def value(self):
        v = self._var.value
        if v is None:
            return None
        return np.asarray(v).flatten() + s_goal


def _baseline_factory(n):
    wrapper = _CartpoleBaselineProb(n)
    return wrapper, wrapper._s_actual, wrapper._u


def _baseline_sweep_step(args, results_dir, method_results, loss_results):
    if args.baseline:
        print("Running baseline sweep...")
        plot_baseline_sweep(results_dir, show=args.show)


if __name__ == "__main__":
    run_plot_main(
        SPEC, METHOD_CONFIGS,
        trajectory_plot_fn=plot_cartpole_trajectory,
        baseline_factory=_baseline_factory,
        baseline_n_fallback=80,
        extra_steps=[_baseline_sweep_step],
    )
