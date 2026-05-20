#!/usr/bin/env python
"""Profile cvxpylayer (ZOH) creation for the stiff_sys example.

For each ``(n_s, n_c)`` cell we:
  * time four phases of problem creation (vars/params, expr build, problem
    init, layer init) using ``perf_counter``;
  * wrap the whole build in ``cProfile`` and dump a ``.prof`` plus the top-20
    cumulative-time entries;
  * run one Adam epoch through the resulting layer so we can compare creation
    vs. per-epoch cost.

Outputs (relative to this script's directory):
  ``bench/creation_breakdown.csv``         -- one row per cell
  ``bench/profiles/zoh_create_ns{n_s}_nc{n_c}.prof``

This machine is RAM-limited (8 GB). The default grid is sized to stay well
under that ceiling with ``backward_mode="lsqr"``. The existing dense-mode
sweep (csv/stiff_sys_solve_times_sweep__dense.csv) hits 34 GB RSS at
(n_s=20, n_c=60) and must NOT be reproduced here.
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from cvxpylayers.torch import CvxpyLayer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "stiff_sys"))

from stiff_sys_times import build_stiff_system  # noqa: E402

from dito import ZOHParamSpec, theta_2_dt, zoh_cost_matrices  # noqa: E402


BENCH_DIR = os.path.join(THIS_DIR, "bench")
PROFILE_DIR = os.path.join(BENCH_DIR, "profiles")
CSV_PATH = os.path.join(BENCH_DIR, "creation_breakdown.csv")

# RAM-safe grid for an 8 GB machine in lsqr mode. The largest cell (10, 60)
# stays around 1 GB RSS based on the existing dense-mode sweep extrapolation
# (lsqr is much lighter than dense).
SMALL_GRID = [(3, 20), (3, 40), (5, 20), (5, 40), (5, 60), (10, 20), (10, 40)]
TINY_GRID = [(3, 10), (5, 20)]

T_HORIZON = 10.0
U_MAX = 10.0


def build_zoh_problem_instrumented(n_s, n_c):
    """Inline ``create_stiff_sys_zoh_clqr`` with phase timing markers.

    Returns (timings, layer, spec, s0_np, sys_mats).
    """
    A_np, B_np, Q_np, R_np, s0_np, _, n_u = build_stiff_system(n_s)
    sys_mats = (A_np, B_np, Q_np, R_np)

    t0 = time.perf_counter()
    spec = ZOHParamSpec(A_np, B_np, Q_np, R_np)
    s_vars = [s0_np] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n_c)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n_c)]
    step_params = spec.make_step_params(n_c)
    t1 = time.perf_counter()

    objective = spec.total_cost_expr(s_vars[:n_c], u_vars, step_params)
    S_next = cp.vstack(s_vars[1:])
    U_mat = cp.vstack(u_vars)
    DYN = cp.vstack([
        spec.dynamics_expr(s_vars[k], u_vars[k], step_params[k])
        for k in range(n_c)
    ])
    constraints = [S_next == DYN, cp.abs(U_mat) <= U_MAX]
    t2 = time.perf_counter()

    problem = cp.Problem(cp.Minimize(objective), constraints)
    t3 = time.perf_counter()

    layer = CvxpyLayer(
        problem,
        parameters=spec.layer_parameters(step_params),
        variables=s_vars[1:] + u_vars,
        canon_backend=cp.COO_CANON_BACKEND,
    )
    t4 = time.perf_counter()

    timings = {
        "phase_vars_params": t1 - t0,
        "phase_expr_build": t2 - t1,
        "phase_problem_init": t3 - t2,
        "phase_layer_init": t4 - t3,
        "dQP_create_total": t4 - t0,
    }
    return timings, layer, spec, s0_np, sys_mats


def time_one_epoch(layer, spec, s0_np, sys_mats, n_c, dtype=torch.float32):
    """One Adam-step round-trip: pack -> forward -> loss -> backward."""
    A_t = torch.tensor(sys_mats[0], dtype=dtype)
    B_t = torch.tensor(sys_mats[1], dtype=dtype)
    Q_t = torch.tensor(sys_mats[2], dtype=dtype)
    R_t = torch.tensor(sys_mats[3], dtype=dtype)

    theta = torch.nn.Parameter(torch.ones(n_c, 1, dtype=dtype))
    solver_args = {"mode": "lsqr", "solve_method": "CLARABEL"}

    dts_torch = theta_2_dt(theta, T_HORIZON, n_c)

    t0 = time.perf_counter()
    packed_steps, W_list = [], []
    for k in range(n_c):
        Ad_k, Bd_k, W_k = zoh_cost_matrices(dts_torch[k], A_t, B_t, Q_t, R_t)
        W_list.append(W_k)
        packed_steps.append(spec.pack_step(Ad_k, Bd_k, W_k))
    t1 = time.perf_counter()

    sol = layer(*spec.flatten_for_layer(packed_steps), solver_args=solver_args)
    t2 = time.perf_counter()

    s0_t = torch.tensor(s0_np, dtype=dtype)
    loss = torch.tensor(0.0, dtype=dtype)
    for k in range(n_c):
        s_k = s0_t if k == 0 else sol[k - 1].to(dtype)
        u_k = sol[n_c + k].to(dtype)
        z_k = torch.cat([s_k, u_k])
        loss = loss + z_k @ W_list[k] @ z_k
    t3 = time.perf_counter()

    loss.backward()
    t4 = time.perf_counter()

    return {
        "epoch_packstep": t1 - t0,
        "epoch_layer_forward": t2 - t1,
        "epoch_loss": t3 - t2,
        "epoch_backward": t4 - t3,
        "epoch_total": t4 - t0,
    }


def profile_one_cell(n_s, n_c, profile=True):
    """Build the problem (optionally under cProfile) and return artifacts."""
    if profile:
        profile_path = os.path.join(
            PROFILE_DIR, f"zoh_create_ns{n_s}_nc{n_c}.prof"
        )
        profiler = cProfile.Profile()
        profiler.enable()
        timings, layer, spec, s0_np, sys_mats = build_zoh_problem_instrumented(
            n_s, n_c
        )
        profiler.disable()
        profiler.dump_stats(profile_path)

        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
        top20 = buf.getvalue()
    else:
        timings, layer, spec, s0_np, sys_mats = build_zoh_problem_instrumented(
            n_s, n_c
        )
        top20 = ""

    return timings, layer, spec, s0_np, sys_mats, top20


def run_sweep(grid, *, do_profile=True, do_epoch=True, n_epoch_warmup=1):
    """Iterate the grid and emit a row per cell.

    A small (3, 5) warm-up is run before the first profiled cell so JIT-style
    one-time imports don't pollute the first measurement.
    """
    print("Warm-up build at (n_s=3, n_c=5) ...", flush=True)
    _ = build_zoh_problem_instrumented(3, 5)

    rows = []
    for n_s, n_c in grid:
        print(f"\n=== n_s={n_s}, n_c={n_c} ===", flush=True)

        ctimings, layer, spec, s0_np, sys_mats, top20 = profile_one_cell(
            n_s, n_c, profile=do_profile,
        )
        print("  Phase wall-times (s):")
        for k, v in ctimings.items():
            print(f"    {k:<22}: {v:.4f}")
        if do_profile:
            print("  --- cProfile top 20 (cumulative) ---")
            print(top20)

        row = {"n_states": n_s, "n_control": n_c, **ctimings}

        if do_epoch:
            for _ in range(n_epoch_warmup):
                _ = time_one_epoch(layer, spec, s0_np, sys_mats, n_c)
            etimings = time_one_epoch(layer, spec, s0_np, sys_mats, n_c)
            print("  Epoch timings (s):")
            for k, v in etimings.items():
                print(f"    {k:<22}: {v:.4f}")
            row.update(etimings)

        rows.append(row)

        # Drop the layer/spec before the next cell to avoid stacking RAM use.
        del layer, spec, s0_np, sys_mats

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Profile dQP creation for the stiff_sys ZOH builder. RAM-safe "
            "defaults; override with --ns / --nc only if you have headroom."
        )
    )
    parser.add_argument(
        "--grid", choices=["small", "tiny"], default="small",
        help="small (default): up to (10, 40); tiny: 2-cell smoke test."
    )
    parser.add_argument(
        "--ns", type=int, nargs="+", default=None,
        help="Explicit n_states list (overrides --grid)."
    )
    parser.add_argument(
        "--nc", type=int, nargs="+", default=None,
        help="Explicit n_control list (overrides --grid)."
    )
    parser.add_argument(
        "--no-profile", action="store_true",
        help="Skip cProfile (faster, no .prof dumps)."
    )
    parser.add_argument(
        "--no-epoch", action="store_true",
        help="Skip the per-epoch timing tail."
    )
    args = parser.parse_args()

    os.makedirs(PROFILE_DIR, exist_ok=True)

    if args.ns is not None and args.nc is not None:
        grid = [(ns, nc) for ns in args.ns for nc in args.nc]
    elif args.grid == "tiny":
        grid = TINY_GRID
    else:
        grid = SMALL_GRID

    df = run_sweep(
        grid,
        do_profile=not args.no_profile,
        do_epoch=not args.no_epoch,
    )

    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {CSV_PATH}")
    print(df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
