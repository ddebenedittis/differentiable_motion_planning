#!/usr/bin/env python
"""Stiff system: dQP build time + training time vs (n_states, n_control).

Mirrors ``ocslc/examples/stiff_system_times.py`` for the differentiable
(cvxpylayers, ZOH) method. For each grid cell the script reports:

    - dQP_create     : time to build the cvxpy.Problem and compile the layer
    - train_total    : time to run ``n_epochs`` Adam steps through the layer
    - train_per_epoch: train_total / n_epochs
    - total          : dQP_create + train_total

Use ``--test`` for a tiny smoke run on a 2x2 grid with very few epochs.
"""

import argparse
import ctypes
import os
import resource
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stiff_sys_prob import create_stiff_sys_zoh_clqr  # noqa: E402
from dito import theta_2_dt, zoh_cost_matrices  # noqa: E402

from dimp.utils import init_matplotlib, get_colors  # noqa: E402

# Match OCSLC's stiff_system_times.py: keep |lambda_max| / |lambda_min| constant
# across n_states so the stiffness ratio stays comparable.
LAMBDA_MAX = 10.0
STIFFNESS_RATIO = 1000.0
T_HORIZON = 10.0
U_MAX = 10.0
X_MAX = None

FULL_N_STATES_LIST = [5, 10, 15, 20, 25, 30]
FULL_N_CONTROL_LIST = list(range(20, 201, 20))
FULL_N_EPOCHS = 20

TEST_N_STATES_LIST = [3, 5]
TEST_N_CONTROL_LIST = [10, 20]
TEST_N_EPOCHS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
CSV_FILENAME_TMPL = "stiff_sys_solve_times_sweep__{mode}.csv"

RESULTS_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "results", "stiff_sys"
)
PLOT_TOTAL_FILENAME_TMPL = "total_time_vs_n_control_sweep__{mode}.pdf"
PLOT_COMPONENTS_FILENAME_TMPL = "total_time_components_sweep__{mode}.pdf"


# Force glibc to release free pages back to the OS. Without this, RSS climbs
# monotonically because glibc holds freed allocations in its arenas for reuse.
try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int

    def malloc_trim():
        _libc.malloc_trim(0)
except (OSError, AttributeError):
    def malloc_trim():
        pass


def _rss_mb():
    """Current resident set size of this process in MiB (Linux)."""
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGESIZE") / (1024 * 1024)
    except (OSError, ValueError):
        return float("nan")


def _peak_rss_mb():
    """Peak RSS for the process so far, in MiB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_stiff_system(n_states):
    """Construct an n_states-dim stiff system with constant stiffness ratio."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2 for a stiff system")

    magnitudes = np.geomspace(LAMBDA_MAX, LAMBDA_MAX / STIFFNESS_RATIO, n_states)
    A = np.diag(-magnitudes)
    B = np.ones((n_states, 1))
    Q = np.eye(n_states)
    R = 0.01 * np.eye(1)
    s0 = -np.ones(n_states)
    return A, B, Q, R, s0, n_states, 1


def time_one_cell(n_states, n_control, n_epochs, backward_mode="lsqr", dtype=torch.float32):
    """Time dQP creation and training for one (n_states, n_control) cell."""
    A_np, B_np, Q_np, R_np, s0_np, n_s, n_u = build_stiff_system(n_states)

    A_t = torch.tensor(A_np, dtype=dtype)
    B_t = torch.tensor(B_np, dtype=dtype)
    Q_t = torch.tensor(Q_np, dtype=dtype)
    R_t = torch.tensor(R_np, dtype=dtype)

    rss_start = _rss_mb()

    t0 = time.perf_counter()
    _, layer, _, _, spec = create_stiff_sys_zoh_clqr(
        n_control, s0_np, n_s, n_u, U_MAX, X_MAX,
        A_sys=A_np, B_sys=B_np, Q_sys=Q_np, R_sys=R_np,
    )
    dQP_create = time.perf_counter() - t0

    rss_after_create = _rss_mb()

    theta = torch.nn.Parameter(torch.ones(n_control, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=1e-2)

    solver_args = {"mode": backward_mode, "solve_method": "CLARABEL"}

    t0 = time.perf_counter()
    for _ in range(n_epochs):
        optim.zero_grad(set_to_none=True)

        dts_torch = theta_2_dt(theta, T_HORIZON, n_control)

        packed_steps, W_list = [], []
        for k in range(n_control):
            Ad_k, Bd_k, W_k = zoh_cost_matrices(
                dts_torch[k], A_t, B_t, Q_t, R_t,
            )
            W_list.append(W_k)
            packed_steps.append(spec.pack_step(Ad_k, Bd_k, W_k))

        sol = layer(
            *spec.flatten_for_layer(packed_steps),
            solver_args=solver_args,
        )

        s0_t = torch.tensor(s0_np, dtype=dtype)
        loss = torch.tensor(0.0, dtype=dtype)
        for k in range(n_control):
            s_k = s0_t if k == 0 else sol[k - 1].to(dtype)
            u_k = sol[n_control + k].to(dtype)
            z_k = torch.cat([s_k, u_k])
            loss = loss + z_k @ W_list[k] @ z_k

        loss.backward()
        optim.step()

        malloc_trim()
    train_total = time.perf_counter() - t0

    rss_end = _rss_mb()
    rss_peak = _peak_rss_mb()

    return {
        "dQP_create": dQP_create,
        "train_total": train_total,
        "train_per_epoch": train_total / n_epochs,
        "total": dQP_create + train_total,
        "rss_start_mb": rss_start,
        "rss_after_create_mb": rss_after_create,
        "rss_end_mb": rss_end,
        "rss_peak_mb": rss_peak,
    }


def parse_int_range(spec):
    """Parse "start:stop[:step]" (Python range semantics), a comma list, or a single int."""
    s = spec.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                f"Expected 'start:stop[:step]', got {spec!r}"
            )
        try:
            nums = [int(p) for p in parts]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Non-integer in range {spec!r}"
            ) from e
        return list(range(*nums))
    if "," in s:
        return [int(p) for p in s.split(",")]
    return [int(s)]


def run_sweep(n_states_list, n_control_list, n_epochs, backward_mode="lsqr"):
    rows = []
    pbar = tqdm(
        total=len(n_states_list) * len(n_control_list),
        desc="Sweeping (n_s, n_c)",
        unit="pt",
    )
    for n_states in n_states_list:
        for n_control in n_control_list:
            timing = time_one_cell(
                n_states, n_control, n_epochs, backward_mode=backward_mode,
            )
            rows.append({
                "n_states": n_states,
                "n_control": n_control,
                **timing,
            })
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def plot_total(df, *, figsize=(3.8, 2.8)):
    figure, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    colors = get_colors()
    for i, n_states in enumerate(sorted(df["n_states"].unique())):
        sub = df[df["n_states"] == n_states].sort_values("n_control")
        ax.plot(
            sub["n_control"],
            sub["total"],
            "s-",
            color=colors[i % len(colors)],
            linewidth=1.5,
            markersize=4,
            label=f"$n_s={int(n_states)}$",
        )
    ax.set_xlabel("Number of control steps")
    ax.set_ylabel("Total time [s]")
    ax.legend(fontsize=8, ncol=2)
    return figure


def plot_components(df, *, figsize=(7.2, 2.8)):
    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    colors = get_colors()
    titles = ("dQP creation", "Training")
    cols = ("dQP_create", "train_total")
    for ax, title, col in zip(axes, titles, cols):
        for i, n_states in enumerate(sorted(df["n_states"].unique())):
            sub = df[df["n_states"] == n_states].sort_values("n_control")
            ax.plot(
                sub["n_control"],
                sub[col],
                "s-",
                color=colors[i % len(colors)],
                linewidth=1.5,
                markersize=4,
                label=f"$n_s={int(n_states)}$",
            )
        ax.set_xlabel("Number of control steps")
        ax.set_ylabel(f"{title} time [s]")
        ax.set_title(title)
    axes[-1].legend(fontsize=8, ncol=2)
    return figure


def warmup(backward_mode="lsqr"):
    """Trigger torch / CvxpyLayer first-call overhead so it doesn't bias cell #1."""
    time_one_cell(
        n_states=3, n_control=10, n_epochs=2, backward_mode=backward_mode,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stiff system: dQP build + training time vs (n_states, n_control) "
            "for the differentiable (cvxpylayers, ZOH) method."
        )
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run on a tiny grid (2x2) with very few epochs for a smoke check.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the number of training epochs per grid cell.",
    )
    parser.add_argument(
        "--ns", type=parse_int_range, default=None,
        help=(
            "Override n_states list. Accepts 'start:stop[:step]' (Python range "
            "semantics), a comma-separated list, or a single int."
        ),
    )
    parser.add_argument(
        "--nc", type=parse_int_range, default=None,
        help=(
            "Override n_control list. Accepts 'start:stop[:step]' (Python range "
            "semantics), a comma-separated list, or a single int."
        ),
    )
    parser.add_argument(
        "--plot", type=str, default="display",
        choices=["display", "save", "none"],
        help="How to handle plots after the sweep.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip the sweep and re-render plots from the saved CSV.",
    )
    parser.add_argument(
        "--backward-mode", type=str, default="lsqr",
        choices=["dense", "lsqr"],
        help=(
            "diffcp backward mode. 'dense' factorizes the full KKT Jacobian "
            "(faster per epoch, much heavier RAM); 'lsqr' uses iterative matvecs "
            "(lower peak RAM, slower backward). The mode is stamped into CSV/plot "
            "filenames so two runs don't overwrite each other."
        ),
    )
    args = parser.parse_args()

    if args.test:
        n_states_list = TEST_N_STATES_LIST
        n_control_list = TEST_N_CONTROL_LIST
        n_epochs = args.epochs or TEST_N_EPOCHS
    else:
        n_states_list = FULL_N_STATES_LIST
        n_control_list = FULL_N_CONTROL_LIST
        n_epochs = args.epochs or FULL_N_EPOCHS

    if args.ns is not None:
        n_states_list = args.ns
    if args.nc is not None:
        n_control_list = args.nc
    if not n_states_list or not n_control_list:
        parser.error("--ns and --nc must each expand to at least one value")

    init_matplotlib()

    backward_mode = args.backward_mode

    os.makedirs(CSV_DIR, exist_ok=True)
    csv_path = os.path.join(
        CSV_DIR, CSV_FILENAME_TMPL.format(mode=backward_mode),
    )

    if args.plot_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"--plot-only requires an existing CSV at {csv_path}; "
                "run without --plot-only first."
            )
        df = pd.read_csv(csv_path)
    else:
        print(f"Mode          : {'test' if args.test else 'full'}")
        print(f"backward mode : {backward_mode}")
        print(f"n_states list : {n_states_list}")
        print(f"n_control list: {n_control_list}")
        print(f"epochs/iter   : {n_epochs}")
        print()

        warmup(backward_mode=backward_mode)
        df = run_sweep(
            n_states_list, n_control_list, n_epochs,
            backward_mode=backward_mode,
        )
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 122)
        print(
            f"Total time vs (n_states, n_control) -- differentiable ZOH "
            f"({backward_mode} + CLARABEL)"
        )
        print("=" * 122)
        print(
            f"{'n_states':<10} {'n_control':<10} {'dQP':<12} "
            f"{'train_tot':<12} {'train/ep':<12} {'total':<12} "
            f"{'rss_create':<12} {'rss_end':<12} {'rss_peak':<12}"
        )
        print("-" * 122)
        for _, row in df.iterrows():
            print(
                f"{int(row['n_states']):<10} {int(row['n_control']):<10} "
                f"{row['dQP_create']:<12.4f} {row['train_total']:<12.4f} "
                f"{row['train_per_epoch']:<12.4f} {row['total']:<12.4f} "
                f"{row['rss_after_create_mb']:<12.1f} "
                f"{row['rss_end_mb']:<12.1f} {row['rss_peak_mb']:<12.1f}"
            )
        print("=" * 122 + "\n")

    fig_total = plot_total(df)
    fig_components = plot_components(df)

    if args.plot in ("save", "display"):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig_total.savefig(
            os.path.join(
                RESULTS_DIR,
                PLOT_TOTAL_FILENAME_TMPL.format(mode=backward_mode),
            ),
            bbox_inches="tight",
        )
        fig_components.savefig(
            os.path.join(
                RESULTS_DIR,
                PLOT_COMPONENTS_FILENAME_TMPL.format(mode=backward_mode),
            ),
            bbox_inches="tight",
        )

    if args.plot == "display":
        plt.show()
    else:
        plt.close(fig_total)
        plt.close(fig_components)


if __name__ == "__main__":
    main()
