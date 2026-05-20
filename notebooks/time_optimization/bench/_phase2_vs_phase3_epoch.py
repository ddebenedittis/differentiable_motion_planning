"""Side-by-side Phase 2 vs Phase 3 iteration-timing bench.

Builds the *same* fully_diagonal-tier ZOH problem in two layouts:
    P2: per-step `cp.Parameter` objects (legacy Phase 2 API), CPP canon backend
    P3: stacked N-dim `cp.Parameter` objects (Phase 3 API), COO canon backend

Times pack_step + layer forward + backward at several (n_s, n_c) cells. Both
layers are solved at the same theta (ones), same dtype, same solver settings.

Output: prints a table and writes
    notebooks/time_optimization/bench/epoch_p2_vs_p3.csv
"""
import os
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from cvxpylayers.torch import CvxpyLayer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(TOP_DIR, "time_optimization", "stiff_sys"))

from stiff_sys_times import build_stiff_system  # noqa: E402

from dito import ZOHParamSpec, theta_2_dt, zoh_cost_matrices  # noqa: E402

DTYPE = torch.float64
T = 10.0
U_MAX = 10.0
SOLVER_ARGS = {"mode": "lsqr", "solve_method": "CLARABEL"}


# ------------------------------------------------------------- Phase 2 builder

def build_phase2(A, B, Q, R, s0_np, n_c):
    """Per-step parameters, fully_diagonal tier, default CPP canon backend."""
    n_s = A.shape[0]
    n_u = B.shape[1]

    s_vars = [s0_np] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n_c)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n_c)]

    Ad_l = [cp.Parameter(n_s, name=f"Ad_{k}") for k in range(n_c)]
    Bd_l = [cp.Parameter((n_s, n_u), name=f"Bd_{k}") for k in range(n_c)]
    Ld_l = [cp.Parameter(n_s, name=f"Ld_{k}") for k in range(n_c)]
    Lxu_l = [cp.Parameter((n_s, n_u), name=f"Lxu_{k}") for k in range(n_c)]
    Luu_l = [cp.Parameter((n_u, n_u), name=f"Luu_{k}") for k in range(n_c)]

    pieces = []
    for k in range(n_c):
        state = cp.multiply(Ld_l[k], s_vars[k]) + Lxu_l[k] @ u_vars[k]
        inp = Luu_l[k] @ u_vars[k]
        pieces.append(cp.hstack([state, inp]))
    objective = cp.sum_squares(cp.hstack(pieces))

    S_next = cp.vstack(s_vars[1:])
    U_mat = cp.vstack(u_vars)
    DYN = cp.vstack([
        cp.multiply(Ad_l[k], s_vars[k]) + Bd_l[k] @ u_vars[k]
        for k in range(n_c)
    ])
    constraints = [S_next == DYN, cp.abs(U_mat) <= U_MAX]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    parameters = Ad_l + Bd_l + Ld_l + Lxu_l + Luu_l
    layer = CvxpyLayer(
        problem, parameters=parameters, variables=s_vars[1:] + u_vars,
    )  # default CPP canon backend
    return layer


def time_epoch_phase2(layer, A_t, B_t, Q_t, R_t, s0_t, n_c, n_s, n_u):
    theta = torch.nn.Parameter(torch.ones(n_c, 1, dtype=DTYPE))
    dts = theta_2_dt(theta, T, n_c)

    t0 = time.perf_counter()
    Ad_list, Bd_list, Ld_list, Lxu_list, Luu_list, W_list = [], [], [], [], [], []
    for k in range(n_c):
        Ad_k, Bd_k, W_k = zoh_cost_matrices(dts[k], A_t, B_t, Q_t, R_t)
        W_list.append(W_k)
        L = torch.linalg.cholesky(W_k)
        LT = L.T
        Ad_list.append(torch.diagonal(Ad_k, 0))
        Bd_list.append(Bd_k)
        Ld_list.append(torch.diagonal(LT[:n_s, :n_s], 0))
        Lxu_list.append(LT[:n_s, n_s:])
        Luu_list.append(LT[n_s:, n_s:])
    t1 = time.perf_counter()

    args = Ad_list + Bd_list + Ld_list + Lxu_list + Luu_list
    sol = layer(*args, solver_args=SOLVER_ARGS)
    t2 = time.perf_counter()

    loss = torch.tensor(0.0, dtype=DTYPE)
    for k in range(n_c):
        s_k = s0_t if k == 0 else sol[k - 1].to(DTYPE)
        u_k = sol[n_c + k].to(DTYPE)
        z_k = torch.cat([s_k, u_k])
        loss = loss + z_k @ W_list[k] @ z_k
    t3 = time.perf_counter()

    loss.backward()
    t4 = time.perf_counter()

    return {
        "packstep": t1 - t0,
        "forward":  t2 - t1,
        "loss":     t3 - t2,
        "backward": t4 - t3,
        "total":    t4 - t0,
        "loss_val": float(loss.item()),
    }


# ------------------------------------------------------------- Phase 3 builder

def build_phase3(A, B, Q, R, s0_np, n_c):
    """Stacked parameters via ZOHParamSpec, COO canon backend."""
    spec = ZOHParamSpec(A, B, Q, R)
    n_s = A.shape[0]
    n_u = B.shape[1]

    s_vars = [s0_np] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n_c)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n_c)]
    step_params = spec.make_step_params(n_c)

    objective = spec.total_cost_expr(s_vars[:n_c], u_vars, step_params)
    S_next = cp.vstack(s_vars[1:])
    U_mat = cp.vstack(u_vars)
    DYN = cp.vstack([
        spec.dynamics_expr(s_vars[k], u_vars[k], step_params[k])
        for k in range(n_c)
    ])
    constraints = [S_next == DYN, cp.abs(U_mat) <= U_MAX]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    layer = CvxpyLayer(
        problem,
        parameters=spec.layer_parameters(step_params),
        variables=s_vars[1:] + u_vars,
        canon_backend=cp.COO_CANON_BACKEND,
    )
    return layer, spec


def time_epoch_phase3(layer, spec, A_t, B_t, Q_t, R_t, s0_t, n_c, n_s, n_u):
    theta = torch.nn.Parameter(torch.ones(n_c, 1, dtype=DTYPE))
    dts = theta_2_dt(theta, T, n_c)

    t0 = time.perf_counter()
    packed, W_list = [], []
    for k in range(n_c):
        Ad_k, Bd_k, W_k = zoh_cost_matrices(dts[k], A_t, B_t, Q_t, R_t)
        W_list.append(W_k)
        packed.append(spec.pack_step(Ad_k, Bd_k, W_k))
    args = spec.flatten_for_layer(packed)
    t1 = time.perf_counter()

    sol = layer(*args, solver_args=SOLVER_ARGS)
    t2 = time.perf_counter()

    loss = torch.tensor(0.0, dtype=DTYPE)
    for k in range(n_c):
        s_k = s0_t if k == 0 else sol[k - 1].to(DTYPE)
        u_k = sol[n_c + k].to(DTYPE)
        z_k = torch.cat([s_k, u_k])
        loss = loss + z_k @ W_list[k] @ z_k
    t3 = time.perf_counter()

    loss.backward()
    t4 = time.perf_counter()

    return {
        "packstep": t1 - t0,
        "forward":  t2 - t1,
        "loss":     t3 - t2,
        "backward": t4 - t3,
        "total":    t4 - t0,
        "loss_val": float(loss.item()),
    }


# --------------------------------------------------------------------- driver

def run_cell(n_s, n_c, n_samples=4, warmup=1):
    A, B, Q, R, s0_np, _, n_u = build_stiff_system(n_s)
    A_t = torch.tensor(A, dtype=DTYPE)
    B_t = torch.tensor(B, dtype=DTYPE)
    Q_t = torch.tensor(Q, dtype=DTYPE)
    R_t = torch.tensor(R, dtype=DTYPE)
    s0_t = torch.tensor(s0_np, dtype=DTYPE)

    layer_p2 = build_phase2(A, B, Q, R, s0_np, n_c)
    layer_p3, spec_p3 = build_phase3(A, B, Q, R, s0_np, n_c)

    samples_p2 = []
    samples_p3 = []
    for i in range(n_samples + warmup):
        r2 = time_epoch_phase2(layer_p2, A_t, B_t, Q_t, R_t, s0_t, n_c, n_s, n_u)
        r3 = time_epoch_phase3(layer_p3, spec_p3,
                               A_t, B_t, Q_t, R_t, s0_t, n_c, n_s, n_u)
        if i >= warmup:
            samples_p2.append(r2)
            samples_p3.append(r3)

    def avg(samples, key):
        return float(np.mean([s[key] for s in samples]))

    return {
        "n_s": n_s, "n_c": n_c,
        # Phase 2
        "p2_packstep": avg(samples_p2, "packstep"),
        "p2_forward":  avg(samples_p2, "forward"),
        "p2_backward": avg(samples_p2, "backward"),
        "p2_total":    avg(samples_p2, "total"),
        # Phase 3
        "p3_packstep": avg(samples_p3, "packstep"),
        "p3_forward":  avg(samples_p3, "forward"),
        "p3_backward": avg(samples_p3, "backward"),
        "p3_total":    avg(samples_p3, "total"),
        # Sanity: losses should match
        "loss_p2": samples_p2[0]["loss_val"],
        "loss_p3": samples_p3[0]["loss_val"],
    }


def main():
    grid = [(3, 20), (3, 40), (5, 20), (5, 40), (5, 60), (10, 20), (10, 40), (10, 60)]
    rows = []
    for n_s, n_c in grid:
        print(f"\n=== n_s={n_s}, n_c={n_c} ===", flush=True)
        row = run_cell(n_s, n_c)
        rows.append(row)
        print(f"  P2 total {row['p2_total']*1000:7.1f} ms "
              f"(pack {row['p2_packstep']*1000:5.1f} fwd {row['p2_forward']*1000:5.1f} "
              f"bwd {row['p2_backward']*1000:5.1f})")
        print(f"  P3 total {row['p3_total']*1000:7.1f} ms "
              f"(pack {row['p3_packstep']*1000:5.1f} fwd {row['p3_forward']*1000:5.1f} "
              f"bwd {row['p3_backward']*1000:5.1f})")
        delta = 100.0 * (row['p3_total'] - row['p2_total']) / row['p2_total']
        print(f"  P3 vs P2 total: {delta:+.1f}%   "
              f"(loss match: {abs(row['loss_p2']-row['loss_p3']) < 1e-8})")

    df = pd.DataFrame(rows)
    out = os.path.join(THIS_DIR, "epoch_p2_vs_p3.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
