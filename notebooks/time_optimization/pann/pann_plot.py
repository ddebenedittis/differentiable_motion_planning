#!/usr/bin/env python
"""Visualization for Pannocchia CLQR differentiable time optimization.

Loads pickles from data/pann_clqr_dt/ and produces plots in results/pann_clqr_dt/.

Usage:
    python pann_plot.py
    python pann_plot.py --method rep zoh
    python pann_plot.py --loss L_IV L_FI
    python pann_plot.py --analysis-only
    python pann_plot.py --show
    python pann_plot.py --save-video
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from pann_prob import (
    create_pann_clqr,
    A, B, s0, T, Q, R, u_max, n_s, n_u,
)
from pann_train import SPEC
from utils import (
    MethodConfig,
    load_pickle,
    pickle_name,
    run_plot_main,
    save_timesteps_video,
)


N = 1000


METHOD_CONFIGS = [
    MethodConfig("aux", "sol_aux", "history_aux",
                 ("time scaled", "time bar scaled"), sol_method=1),
    MethodConfig("rep", "sol_rep", "history_rep", ("time scaled",)),
    MethodConfig("hs", "sol_hs", "history_hs",
                 ("uniform_resample", "substeps")),
    MethodConfig("zoh", "sol_zoh", "history_zoh",
                 ("exact_zoh_integrated",), uses_custom_n=True),
]


_PREFIX_MAP = {"aux": "Aux", "rep": "Rep", "hs": "HS", "zoh": "ZOH"}


# ============================================================================ #
# HS Comparison
# ============================================================================ #

def plot_hs_comparison(result, n, results_dir, show=False):
    history_hs = result["history"]
    methods = result["internal_methods"]

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4), constrained_layout=True)

    for method in methods:
        h_m = [h for h in history_hs if h['method'] == method]
        if not h_m:
            continue
        axs[0, 0].plot([h['loss'] for h in h_m], label=method)
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title("Loss Convergence")
    axs[0, 0].legend()

    for method in methods:
        h_m = [h for h in history_hs if h['method'] == method]
        if not h_m:
            continue
        d_arr = h_m[-1]['dts']
        axs[0, 1].plot(np.cumsum(d_arr), d_arr, label=method)
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Timestep duration")
    axs[0, 1].set_title("Final Timestep Distributions")
    axs[0, 1].legend()

    for i, method in enumerate(methods[:2]):
        h_m = [h for h in history_hs if h['method'] == method]
        if not h_m:
            continue
        d_arr = h_m[-1]['dts']
        axs[1, i].hist(d_arr.flatten(), bins=30, alpha=0.7, edgecolor='black')
        axs[1, i].set_xlabel("Timestep duration")
        axs[1, i].set_ylabel("Count")
        axs[1, i].set_title(f"Histogram: {method}")
        axs[1, i].axvline(T / n, color='r', linestyle='--',
                          label=f"uniform={T/n:.4f}")
        axs[1, i].legend()

    fig.suptitle("Comparison: Uniform Resampling vs Substeps", fontsize=14)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "hs_comparison.pdf"),
                    bbox_inches='tight')
    if not show:
        plt.close(fig)


def _hs_comparison_step(args, results_dir, method_results, loss_results):
    if "hs" in method_results:
        print("Plotting HS comparison...")
        plot_hs_comparison(method_results["hs"], SPEC.n_default, results_dir,
                           show=args.show)


# ============================================================================ #
# Baseline Sweep
# ============================================================================ #

def plot_baseline_sweep(results_dir, show=False):
    interval = 80
    fig, axs = plt.subplots(12, 2, figsize=(6.4, 12.8),
                            constrained_layout=True)

    i = 0
    for n_test in range(interval, N + 1, interval):
        dt_test = T / n_test
        prob_test, s_test, u_test = create_pann_clqr(
            n_test, s0, A, B, Q, R, dt_test, u_max)
        t0 = time.time()
        prob_test.solve()
        solve_time = time.time() - t0
        print(f"n = {n_test}\nOptimal cost: {prob_test.objective.value:.4f}, "
              f"solve time meas: {solve_time:.4f} s, "
              f"solve time: {prob_test.solver_stats.solve_time:.4f} s\n")

        times = np.arange(n_test) * dt_test
        s_vec = np.array([sv.value for sv in s_test[1:n_test + 1]])
        u_vec = np.array([uv.value for uv in u_test[:n_test]])

        axs[2 * (i // 2), i % 2].plot(times, s_vec, label=['x', 'y', 'z'])
        axs[2 * (i // 2), i % 2].set(xlabel='Time', ylabel='State',
                                     title=fr"$n={n_test}$")
        axs[2 * (i // 2) + 1, i % 2].plot(times, u_vec)
        axs[2 * (i // 2) + 1, i % 2].set(xlabel='Time', ylabel='Input')
        i += 1

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "baseline_sweep.pdf"),
                    bbox_inches='tight')
    if not show:
        plt.close(fig)


def _baseline_step(args, results_dir, method_results, loss_results):
    if args.baseline:
        print("Running baseline sweep...")
        plot_baseline_sweep(results_dir, show=args.show)


# ============================================================================ #
# Timestep Evolution Videos
# ============================================================================ #

_VIDEO_CONFIG = {
    ("aux", "time scaled"):
        ("dts_dist_aux_time_scaled", "aux_time_scaled_timesteps"),
    ("aux", "time bar scaled"):
        ("dts_dist_aux_time_bar_scaled", "aux_time_bar_scaled_timesteps"),
    ("rep", "time scaled"):
        ("dts_dist_rep_time_scaled", "rep_timesteps"),
    ("hs", "uniform_resample"):
        ("dts_dist_hs_uniform_resample", "hs_uniform_timesteps"),
    ("hs", "substeps"):
        ("dts_dist_hs_substeps", "hs_substeps_timesteps"),
}


def _videos_step(args, results_dir, method_results, loss_results):
    if not args.save_video:
        return

    print("Saving timestep evolution videos...")
    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", SPEC.data_dirname)
    n_default = SPEC.n_default

    for name, result in method_results.items():
        for method in result["internal_methods"]:
            if name == "zoh":
                n_zoh = result["n"]
                dist_pkl = pickle_name("dts_dist_zoh", n_zoh, n_default)
                video_name = (f"zoh_timesteps_n{n_zoh}"
                              if n_zoh != n_default else "zoh_timesteps")
            else:
                key = (name, method)
                if key not in _VIDEO_CONFIG:
                    continue
                dist_pkl, video_name = _VIDEO_CONFIG[key]

            try:
                dts_all = load_pickle(data_dir, dist_pkl)
            except (FileNotFoundError, OSError):
                print(f"  No dts distribution for {name}/{method}, skipping")
                continue
            print(f"  Video: {video_name}")
            save_timesteps_video(results_dir, video_name, dts_all=dts_all)

    for loss_name in loss_results:
        try:
            dts_dist = load_pickle(data_dir, f"dts_dist_{loss_name}")
        except (FileNotFoundError, OSError):
            print(f"  No dts distribution for {loss_name}, skipping")
            continue
        print(f"  Video: {loss_name}")
        save_timesteps_video(results_dir, f"{loss_name}_timesteps",
                             dts_all=dts_dist)


def _add_pann_args(parser):
    parser.add_argument("--n", type=int, default=SPEC.n_default,
                        help="Default n for pickle naming")
    parser.add_argument("--n-zoh", type=int, default=SPEC.default_n["zoh"],
                        help="ZOH timestep count")
    parser.add_argument("--save-video", action="store_true",
                        help="Save timestep evolution videos")


def _load_method_kwargs(args):
    return {"n_default": args.n, "n_custom": args.n_zoh}


if __name__ == "__main__":
    run_plot_main(
        SPEC, METHOD_CONFIGS,
        parser_extras=_add_pann_args,
        load_method_kwargs_fn=_load_method_kwargs,
        build_solutions_kwargs={
            "label_with_n": True,
            "n_default": SPEC.n_default,
            "prefix_map": _PREFIX_MAP,
        },
        extra_steps=[_hs_comparison_step, _baseline_step, _videos_step],
    )
