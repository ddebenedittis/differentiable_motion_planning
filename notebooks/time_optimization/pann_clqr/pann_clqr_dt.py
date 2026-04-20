#!/usr/bin/env python
"""Data generation script for differentiable time optimization experiments.

Trains 5 optimization methods (aux, rep, hs_uniform, hs_substeps, zoh)
for learning non-uniform timesteps in constrained LQR problems.

Usage:
    python pann_clqr_dt.py --mode full --method rep zoh
    python pann_clqr_dt.py --mode test --method all
    python pann_clqr_dt.py --mode full --method zoh --n 80 --epochs 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from pann_clqr import (
    create_pann_param_clqr,
    create_pann_param_clqr_2,
    create_exact_zoh_cost_clqr,
)
from utils import (
    RunMode,
    get_n_epochs,
    pickle_name,
    get_reparam_fn,
    REPARAM_CHOICES,
    Ad_Bd_from_dt,
    zoh_cost_matrices,
    task_loss,
    uniform_resampling_loss,
    substep_loss,
    save_pickle,
    save_dts_distribution,
)

# ============================================================================ #
# System Constants (Pannocchia)
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

ALL_METHODS = ["aux", "rep", "hs_uniform", "hs_substeps", "zoh"]

DEFAULT_N = {
    "aux": 160, "rep": 160, "hs_uniform": 160, "hs_substeps": 160,
    "zoh": 80,
}
DEFAULT_LR = {
    "aux": 5e-4, "rep": 1e-2, "hs_uniform": 1e-2, "hs_substeps": 1e-2,
    "zoh": 1e-2,
}


# ============================================================================ #
# Aux Training (structurally different from softmax methods)
# ============================================================================ #

def train_aux(n, n_epochs, lr, data_dir):
    """Train the auxiliary variable method.

    Returns:
        sol_aux: dict mapping loss method name -> solution tensors
        history_aux: list of history dicts
    """
    loss_methods = ["time scaled", "time bar scaled"]
    history_aux = []
    sol_aux = {}

    for method in loss_methods:
        print(f"  Aux sub-method: {method}")

        dt_init = T / n
        dts_torch = [torch.nn.Parameter(torch.ones(1) * dt_init) for _ in range(n)]
        optim = torch.optim.Adam(dts_torch, lr=lr)

        with torch.no_grad():
            for d in dts_torch:
                d.copy_(torch.ones(1) * dt_init)

        s_bar = [s0 + (np.zeros(n_s) - s0) * i / n for i in range(n)]
        u_bar = [np.zeros(n_u) for _ in range(n)]

        with tqdm(total=n_epochs, desc=f"aux ({method})") as pbar:
            for epoch in range(n_epochs):
                pbar.update(1)
                optim.zero_grad()

                _, layer_aux, _, _, _, dts_params = create_pann_param_clqr(
                    n, s0, A, B, Q, R, s_bar, u_bar, u_max, T,
                )
                sol_aux[method] = layer_aux(*dts_torch)

                s_bar = [sol_aux[method][i].detach().numpy() for i in range(n)]
                u_bar = [sol_aux[method][n + i].detach().numpy() for i in range(n)]

                states_sol = [sol_aux[method][i] for i in range(n)]
                inputs_sol = [sol_aux[method][n + i] for i in range(n)]
                deltas_sol = [sol_aux[method][2 * n + i] for i in range(n)]

                if method == "time scaled":
                    loss = sum(
                        deltas_sol[i] * states_sol[i].t() @ torch.tensor(Q, dtype=torch.float64) @ states_sol[i]
                        for i in range(n)
                    ) + sum(
                        deltas_sol[i] * inputs_sol[i].t() @ torch.tensor(R, dtype=torch.float64) @ inputs_sol[i]
                        for i in range(n)
                    )
                elif method == "time bar scaled":
                    deltas_bar = np.concatenate([d.detach().numpy() for d in dts_torch])
                    loss = sum(
                        deltas_bar[i] * states_sol[i].t() @ torch.tensor(Q, dtype=torch.float64) @ states_sol[i]
                        for i in range(n)
                    ) + sum(
                        deltas_bar[i] * inputs_sol[i].t() @ torch.tensor(R, dtype=torch.float64) @ inputs_sol[i]
                        for i in range(n)
                    )
                else:
                    loss = task_loss(states_sol, inputs_sol, dts_torch, Q, R, method="unscaled")

                loss.backward()
                optim.step()

                with torch.no_grad():
                    for d in dts_torch:
                        d.clamp_(min=1e-6, max=0.07)
                        d *= T / sum(dts_torch)

                history_aux.append({
                    'method': method,
                    'loss': loss.item(),
                    'dts': [d.detach().numpy() for d in dts_torch],
                })

    # Save results
    save_pickle(data_dir, "sol_aux", sol_aux)
    save_pickle(data_dir, "history_aux", history_aux)
    for method in loss_methods:
        hist_m = [h for h in history_aux if h['method'] == method]
        _key = method.replace(' ', '_')
        save_dts_distribution(data_dir, f"dts_dist_aux_{_key}", hist_m)

    return sol_aux, history_aux


# ============================================================================ #
# Softmax-Based Training (rep, hs_uniform, hs_substeps, zoh)
# ============================================================================ #

def _create_layer(method_name, n):
    """Create the CvxpyLayer for a given method."""
    if method_name in ("rep", "hs_uniform", "hs_substeps"):
        _, layer, _, _, _ = create_pann_param_clqr_2(n, s0, A, B, Q, R, u_max)
        return layer
    elif method_name == "zoh":
        _, layer, _, _, _, _, _, _ = create_exact_zoh_cost_clqr(n, s0, n_s, n_u, u_max)
        return layer
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _compute_qp_params_and_solve(method_name, layer, dts_torch, n, A_t, B_t, Q_t, R_t):
    """Compute QP parameters from dts and solve via the layer.

    Returns:
        sol: layer output tuple
        W_list: list of cost matrices (only for zoh, else None)
    """
    if method_name in ("rep", "hs_uniform", "hs_substeps"):
        return layer(dts_torch), None

    elif method_name == "zoh":
        Ad_list, Bd_list, Lx_list, Lu_list, W_list = [], [], [], [], []
        for k in range(n):
            Ad_k, Bd_k, W_k = zoh_cost_matrices(dts_torch[k], A_t, B_t, Q_t, R_t)
            Ad_list.append(Ad_k)
            Bd_list.append(Bd_k)
            W_list.append(W_k)
            L_k = torch.linalg.cholesky(W_k)
            LT_k = L_k.T
            Lx_list.append(LT_k[:, :n_s])
            Lu_list.append(LT_k[:, n_s:])
        return layer(*Ad_list, *Bd_list, *Lx_list, *Lu_list), W_list

    raise ValueError(f"Unknown method: {method_name}")


def _compute_loss(method_name, sol, dts_torch, n, W_list, s0_t, A_t, B_t, Q_t, R_t):
    """Compute loss for a given method."""
    # CvxpyLayer may return float64; cast to match dts dtype for consistency
    dtype = dts_torch.dtype

    if method_name == "rep":
        states_sol = [sol[i].to(dtype) for i in range(n)]
        inputs_sol = [sol[n + i].to(dtype) for i in range(n)]
        return task_loss(states_sol, inputs_sol, dts_torch, Q, R, method="time_scaled")

    elif method_name == "hs_uniform":
        inputs_qp = [sol[n + i].to(dtype) for i in range(n)]
        return uniform_resampling_loss(
            inputs_qp, dts_torch, s0_t, A_t, B_t, Q_t, R_t,
            T=T, n_res=1000, use_exact=False,
        )

    elif method_name == "hs_substeps":
        inputs_qp = [sol[n + i].to(dtype) for i in range(n)]
        return substep_loss(
            inputs_qp, dts_torch, s0_t, A_t, B_t, Q_t, R_t,
            n_sub=10, use_exact=False,
        )

    elif method_name == "zoh":
        s0_t_local = s0_t
        loss = torch.tensor(0.0, dtype=dtype)
        for k in range(n):
            s_k = s0_t_local if k == 0 else sol[k - 1].to(dtype)
            u_k = sol[n + k].to(dtype)
            z_k = torch.cat([s_k, u_k])
            loss = loss + z_k @ W_list[k] @ z_k
        return loss

    raise ValueError(f"Unknown method: {method_name}")


def _pickle_names(method_name, n, n_default):
    """Return (sol_name, history_name, dts_dist_name) for a method."""
    if method_name == "rep":
        return "sol_rep", "history_rep", "dts_dist_rep_time_scaled"
    elif method_name == "hs_uniform":
        return "sol_hs", "history_hs", "dts_dist_hs_uniform_resample"
    elif method_name == "hs_substeps":
        return "sol_hs", "history_hs", "dts_dist_hs_substeps"
    elif method_name == "zoh":
        sol_name = pickle_name("sol_zoh", n, n_default)
        hist_name = pickle_name("history_zoh", n, n_default)
        dist_name = pickle_name("dts_dist_zoh", n, n_default)
        return sol_name, hist_name, dist_name
    raise ValueError(f"Unknown method: {method_name}")


# Mapping from method_name to internal loss method key used in history/sol dicts
_INTERNAL_METHOD_KEY = {
    "rep": "time scaled",
    "hs_uniform": "uniform_resample",
    "hs_substeps": "substeps",
    "zoh": "exact_zoh_integrated",
}


def train_softmax_method(method_name, n, n_epochs, lr, data_dir, n_default=160,
                         reparam="softmax"):
    """Train a softmax-based method.

    Args:
        method_name: one of "rep", "hs_uniform", "hs_substeps", "zoh"
        n: number of timesteps
        n_epochs: number of training epochs
        lr: learning rate
        data_dir: directory for pickle output
        n_default: default n for pickle naming (used by zoh)
        reparam: reparametrization ("softmax" or "logsoftmax")

    Returns:
        sol: solution dict or tensors
        history: list of history dicts
    """
    internal_key = _INTERNAL_METHOD_KEY[method_name]
    theta_2_dt = get_reparam_fn(reparam)

    # Use float32 for most methods, consistent with notebook
    dtype = torch.float32
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    s0_t = torch.tensor(s0, dtype=dtype)

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    layer = _create_layer(method_name, n)

    history = []
    sol = None
    sol_dict = {}

    with tqdm(total=n_epochs, desc=method_name) as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            dts_torch = theta_2_dt(theta, T, n)
            sol_raw, W_list = _compute_qp_params_and_solve(
                method_name, layer, dts_torch, n, A_t, B_t, Q_t, R_t,
            )
            sol_dict[internal_key] = sol_raw

            loss = _compute_loss(
                method_name, sol_raw, dts_torch, n, W_list, s0_t, A_t, B_t, Q_t, R_t,
            )
            loss.backward()
            optim.step()

            history.append({
                'method': internal_key,
                'epoch': epoch,
                'loss': float(loss.item()),
                'dts': dts_torch.detach().cpu().numpy(),
            })

    sol = sol_dict

    # Determine if this method shares pickles with another (hs_uniform/hs_substeps)
    sol_name, hist_name, dist_name = _pickle_names(method_name, n, n_default)

    # Add suffix for non-default reparametrization
    if reparam != "softmax":
        sol_name += f"_{reparam}"
        hist_name += f"_{reparam}"
        dist_name += f"_{reparam}"

    # For hs methods, we need to handle shared pickle files
    if method_name in ("hs_uniform", "hs_substeps"):
        # Try to load existing sol/history to merge
        try:
            from utils import load_pickle
            existing_sol = load_pickle(data_dir, sol_name)
            existing_hist = load_pickle(data_dir, hist_name)
            existing_sol[internal_key] = sol_dict[internal_key]
            existing_hist.extend(history)
            sol = existing_sol
            history_to_save = existing_hist
        except (FileNotFoundError, OSError):
            history_to_save = history
    else:
        history_to_save = history

    save_pickle(data_dir, sol_name, sol)
    save_pickle(data_dir, hist_name, history_to_save)
    save_dts_distribution(data_dir, dist_name, history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Data generation for differentiable time optimization experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["test", "full"],
        help="Run mode: test (5 epochs) or full (default epochs per method)",
    )
    parser.add_argument(
        "--method", nargs="+", required=True,
        help=f"Method(s) to train, or 'all'. Available: {ALL_METHODS}",
    )
    parser.add_argument("--n", type=int, default=None, help="Override timestep count")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle output directory (default: data/pann_clqr_dt)",
    )
    parser.add_argument(
        "--reparam", default="both",
        choices=[*REPARAM_CHOICES, "both"],
        help="Reparametrization: softmax, logsoftmax, or both (default: both)",
    )
    args = parser.parse_args()

    methods = ALL_METHODS if "all" in args.method else args.method
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method: {m}. Available: {ALL_METHODS}")

    run_mode = RunMode.TEST if args.mode == "test" else RunMode.FULL

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "pann_clqr_dt")
    os.makedirs(data_dir, exist_ok=True)

    # Default n for pickle naming (matches notebook default)
    n_default = 160

    reparams = list(REPARAM_CHOICES) if args.reparam == "both" else [args.reparam]

    print(f"Output directory: {data_dir}")
    print(f"Methods: {methods}")
    print(f"Reparametrizations: {reparams}")
    print(f"Mode: {args.mode}")
    print()

    for reparam in reparams:
        for method_name in methods:
            n = args.n or DEFAULT_N[method_name]
            lr = args.lr or DEFAULT_LR[method_name]
            method_key = "aux" if method_name == "aux" else method_name
            # Map hs methods to the "hs" key for epoch lookup
            epoch_key = method_name
            if method_name in ("hs_uniform", "hs_substeps"):
                epoch_key = "hs"
            n_epochs = args.epochs or get_n_epochs(run_mode, epoch_key)

            print(f"\n{'=' * 40}")
            print(f"Training {method_name} [{reparam}] (n={n}, epochs={n_epochs}, lr={lr})")
            print(f"{'=' * 40}")

            if method_name == "aux":
                train_aux(n, n_epochs, lr, data_dir)
            else:
                train_softmax_method(
                    method_name, n, n_epochs, lr, data_dir,
                    n_default=n_default, reparam=reparam,
                )

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
