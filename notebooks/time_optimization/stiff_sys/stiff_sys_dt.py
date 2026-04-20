#!/usr/bin/env python
"""Data generation script for stiff system LTI differentiable time optimization.

The system has three states with well-separated time constants (0.1s, 10s, 100s),
making it an ideal benchmark for non-uniform timestep placement. The fast mode
(x1) settles quickly while the slow mode (x3) drifts, so placing finer samples
early should significantly improve approximation quality.

Trains methods (rep, zoh) and alternative loss functions for learning
non-uniform timesteps.

Usage:
    python stiff_sys_dt.py --mode test --experiment all
    python stiff_sys_dt.py --mode full --experiment methods --method rep
    python stiff_sys_dt.py --mode full --experiment losses --loss L_IV L_FI
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from stiff_sys_clqr import (
    create_stiff_sys_rep_clqr,
    create_stiff_sys_zoh_clqr,
)
from utils import (
    LOSS_REGISTRY,
    REPARAM_CHOICES,
    AdaptiveGradientBalancer,
    RunMode,
    get_n_epochs,
    get_reparam_fn,
    save_dts_distribution,
    save_pickle,
    task_loss,
    zoh_cost_matrices,
)

# ============================================================================ #
# System Constants (Stiff System LTI)
# ============================================================================ #

A = np.array([
    [-10,     0.0,    0.0],
    [0.0,    -0.1,    0.0],
    [0.0,     0.0,  -0.01],
])

B = np.array([
    [1.0],
    [1.0],
    [1.0],
])

s0 = np.array([-1.0, -1.0, -1.0])
T = 10.0
n_default = 40
Q = 1.0 * np.eye(3)
R = 0.01 * np.eye(1)
u_max = 10.0
x_max = None  # state bounds are very loose (+-100), effectively inactive
n_s = 3
n_u = 1

ALL_METHODS = ["rep", "zoh"]

DEFAULT_N = {"rep": 40, "zoh": 20}
DEFAULT_LR = {"rep": 1e-2, "zoh": 1e-2}


# ============================================================================ #
# Helpers
# ============================================================================ #

def euler_matrices(dt_k, A_t, B_t, Q_t, R_t):
    """Forward Euler discretization + time-scaled block-diagonal cost matrix.

    Returns (Ad, Bd, W) with the same interface as zoh_cost_matrices so the
    training loop can branch without structural changes.
    """
    _n_s = A_t.shape[0]
    _n_u = B_t.shape[1]
    Ad = torch.eye(_n_s, dtype=A_t.dtype) + A_t * dt_k
    Bd = B_t * dt_k
    W = torch.zeros(_n_s + _n_u, _n_s + _n_u, dtype=A_t.dtype)
    W[:_n_s, :_n_s] = Q_t * dt_k
    W[_n_s:, _n_s:] = R_t * dt_k
    return Ad, Bd, W


def save_run_config(data_dir, args):
    """Dump CLI args + git hash to run_config.json."""
    config = vars(args).copy()
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    config["git_hash"] = git_hash
    with open(os.path.join(data_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def build_loss_kwargs(loss_name, states, inputs, dts, W_list, Ad_list, Bd_list,
                      A_t, B_t, Q_t, R_t):
    """Dispatch correct kwargs to each loss function."""
    if loss_name == "L_SSD":
        return dict(dts=dts)
    elif loss_name == "L_IV":
        return dict(inputs=inputs, dts=dts)
    elif loss_name == "L_EQ":
        return dict(inputs=inputs)
    elif loss_name == "L_CPC":
        return dict(inputs=inputs, dts=dts, u_max=u_max)
    elif loss_name == "L_CSS":
        return dict(inputs=inputs, dts=dts, u_max=u_max)
    elif loss_name == "L_defect":
        return dict(states=states, inputs=inputs, dts=dts, W_list=W_list,
                    Q=Q_t, R=R_t)
    elif loss_name == "L_dyn":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t,
                    Ad_list=Ad_list, Bd_list=Bd_list)
    elif loss_name == "L_equi":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t, Q=Q_t)
    elif loss_name == "L_FI":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t, Q=Q_t,
                    T=T)
    elif loss_name == "L_SC":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t,
                    x_max=x_max)
    elif loss_name == "L_PWLH":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t,
                    Q=Q_t, R=R_t)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ============================================================================ #
# Methods Training (rep, zoh) -- Softmax Parametrization
# ============================================================================ #

def _create_layer(method_name, n):
    """Create the CvxpyLayer for a given method."""
    if method_name == "rep":
        _, layer, _, _, _ = create_stiff_sys_rep_clqr(
            n, s0, A, B, Q, R, u_max, x_max,
        )
        return layer
    elif method_name == "zoh":
        _, layer, _, _, _, _, _, _ = create_stiff_sys_zoh_clqr(
            n, s0, n_s, n_u, u_max, x_max,
        )
        return layer
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _compute_qp_params_and_solve(method_name, layer, dts_torch, n, A_t, B_t,
                                  Q_t, R_t):
    """Compute QP parameters from dts and solve via the layer.

    Returns:
        sol: layer output tuple
        W_list: list of cost matrices (only for zoh, else None)
    """
    if method_name == "rep":
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


def _compute_loss(method_name, sol, dts_torch, n, W_list, s0_t):
    """Compute loss for a given method."""
    dtype = dts_torch.dtype

    if method_name == "rep":
        states_sol = [sol[i].to(dtype) for i in range(n)]
        inputs_sol = [sol[n + i].to(dtype) for i in range(n)]
        return task_loss(states_sol, inputs_sol, dts_torch, Q, R,
                         method="time_scaled")

    elif method_name == "zoh":
        loss = torch.tensor(0.0, dtype=dtype)
        for k in range(n):
            s_k = s0_t if k == 0 else sol[k - 1].to(dtype)
            u_k = sol[n + k].to(dtype)
            z_k = torch.cat([s_k, u_k])
            loss = loss + z_k @ W_list[k] @ z_k
        return loss

    raise ValueError(f"Unknown method: {method_name}")


_INTERNAL_METHOD_KEY = {
    "rep": "time_scaled",
    "zoh": "exact_zoh_integrated",
}


def train_softmax_method(method_name, n, n_epochs, lr, data_dir,
                         reparam="softmax"):
    """Train a softmax-based method.

    Args:
        method_name: one of "rep", "zoh"
        n: number of timesteps
        n_epochs: number of training epochs
        lr: learning rate
        data_dir: directory for pickle output
        reparam: reparametrization ("softmax" or "logsoftmax")

    Returns:
        sol: solution dict
        history: list of history dicts
    """
    internal_key = _INTERNAL_METHOD_KEY[method_name]
    theta_2_dt = get_reparam_fn(reparam)

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
                method_name, sol_raw, dts_torch, n, W_list, s0_t,
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

    # Save results
    suffix = f"_{reparam}" if reparam != "softmax" else ""
    sol_name = f"sol_{method_name}{suffix}"
    hist_name = f"history_{method_name}{suffix}"
    dist_name = f"dts_dist_{method_name}{suffix}"

    save_pickle(data_dir, sol_name, sol)
    save_pickle(data_dir, hist_name, history)
    save_dts_distribution(data_dir, dist_name, history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Alt Losses Training (ZOH3 base + regularizer)
# ============================================================================ #

def train_one_loss(loss_name, n, n_epochs, lr, lambda0, use_balancing, data_dir,
                   detach="none", reparam="softmax"):
    """ZOH3 training loop with one alternative loss as regularizer.

    Args:
        detach: Gradient detach mode for the QP solution.
            "none"  -- full gradient through cvxpylayers (default)
            "reg"   -- detach states/inputs for L_reg only
            "all"   -- detach for both L_ocp and L_reg
        reparam: reparametrization ("softmax" or "logsoftmax")

    Returns:
        sol: list of torch tensors (QP solution)
        history: list of dicts with training metrics
    """
    theta_2_dt = get_reparam_fn(reparam)

    dtype = torch.float64
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    s0_t = torch.tensor(s0, dtype=dtype)

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    _, layer, _, _, _, _, _, _ = create_stiff_sys_zoh_clqr(
        n, s0, n_s, n_u, u_max, x_max,
    )

    balancer = AdaptiveGradientBalancer(lambda_0=lambda0) if use_balancing else None
    loss_fn = LOSS_REGISTRY[loss_name]

    history = []
    sol = None

    with tqdm(total=n_epochs, desc=loss_name) as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            dts_torch = theta_2_dt(theta, T, n)

            # Compute discretization parameters (Exact ZOH)
            Ad_list, Bd_list, Lx_list, Lu_list, W_list = [], [], [], [], []
            for k in range(n):
                Ad_k, Bd_k, W_k = zoh_cost_matrices(
                    dts_torch[k], A_t, B_t, Q_t, R_t,
                )
                Ad_list.append(Ad_k)
                Bd_list.append(Bd_k)
                W_list.append(W_k)

                L_k = torch.linalg.cholesky(W_k)
                LT_k = L_k.T
                Lx_list.append(LT_k[:, :n_s])
                Lu_list.append(LT_k[:, n_s:])

            # Solve QP
            sol = layer(*Ad_list, *Bd_list, *Lx_list, *Lu_list)

            # Extract states and inputs
            states = [s0_t] + [sol[k] for k in range(n)]
            inputs = [sol[n + k] for k in range(n)]

            # Detach QP solution from computation graph if requested
            if detach in ("reg", "all"):
                states_d = [s.detach() for s in states]
                inputs_d = [u.detach() for u in inputs]
            states_ocp = states_d if detach == "all" else states
            inputs_ocp = inputs_d if detach == "all" else inputs
            states_reg = states_d if detach in ("reg", "all") else states
            inputs_reg = inputs_d if detach in ("reg", "all") else inputs

            # L_ocp: exact integrated cost
            loss_ocp = torch.tensor(0.0, dtype=dtype)
            for k in range(n):
                z_k = torch.cat([states_ocp[k], inputs_ocp[k]])
                loss_ocp = loss_ocp + z_k @ W_list[k] @ z_k

            # L_reg: alternative loss
            kwargs = build_loss_kwargs(
                loss_name, states_reg, inputs_reg, dts_torch, W_list, Ad_list,
                Bd_list, A_t, B_t, Q_t, R_t,
            )
            loss_reg = loss_fn(**kwargs)

            # Combine losses
            if balancer is not None:
                lambda_hat = balancer.step(theta, loss_ocp, loss_reg)
            else:
                lambda_hat = lambda0

            loss = loss_ocp + lambda_hat * loss_reg
            loss.backward()
            optim.step()

            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_ocp": float(loss_ocp.item()),
                "loss_reg": float(loss_reg.item()),
                "lambda_hat": float(lambda_hat) if isinstance(lambda_hat, float) else float(lambda_hat),
                "dts": dts_torch.detach().cpu().numpy(),
                "detach": detach,
            })

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ocp=f"{loss_ocp.item():.4f}",
                reg=f"{loss_reg.item():.4f}",
                lam=f"{lambda_hat:.4f}" if isinstance(lambda_hat, float) else f"{lambda_hat:.4f}",
            )

    # Save results
    suffix = f"_{reparam}" if reparam != "softmax" else ""
    save_pickle(data_dir, f"sol_{loss_name}{suffix}", sol)
    save_pickle(data_dir, f"history_{loss_name}{suffix}", history)
    save_dts_distribution(data_dir, f"dts_dist_{loss_name}{suffix}", history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Data generation for stiff system LTI differentiable time optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["test", "full"],
        help="Run mode: test (5 epochs) or full (default epochs per method/loss)",
    )
    parser.add_argument(
        "--experiment", required=True, choices=["methods", "losses", "all"],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--method", nargs="+", default=["all"],
        help=f"Method(s) to train, or 'all'. Available: {ALL_METHODS}",
    )
    parser.add_argument(
        "--loss", nargs="+", default=["all"],
        help="Loss function(s) to train, or 'all'. Available: "
             + str(list(LOSS_REGISTRY.keys())),
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Override timestep count")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--lambda0", type=float, default=0.3,
                        help="Base regularizer weight (default: 0.3)")
    parser.add_argument("--no-balancing", action="store_true",
                        help="Use fixed lambda0 instead of adaptive")
    parser.add_argument(
        "--detach", choices=["none", "reg", "all"], default="none",
        help="Detach QP solution from gradient graph: "
             "'none' (default), 'reg' (L_reg only), 'all' (both losses)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle output directory (default: data/stiff_sys_dt)",
    )
    parser.add_argument(
        "--reparam", default="both",
        choices=[*REPARAM_CHOICES, "both"],
        help="Reparametrization: softmax, logsoftmax, or both (default: both)",
    )

    args = parser.parse_args()

    run_mode = RunMode.TEST if args.mode == "test" else RunMode.FULL

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "stiff_sys_dt")
    os.makedirs(data_dir, exist_ok=True)

    reparams = list(REPARAM_CHOICES) if args.reparam == "both" else [args.reparam]

    save_run_config(data_dir, args)

    print(f"Output directory: {data_dir}")
    print(f"Reparametrizations: {reparams}")
    print(f"Mode: {args.mode}")
    print()

    # Methods training
    if args.experiment in ("methods", "all"):
        methods = ALL_METHODS if "all" in args.method else args.method
        for m_name in methods:
            if m_name not in ALL_METHODS:
                parser.error(
                    f"Unknown method: {m_name}. Available: {ALL_METHODS}")

        print(f"Methods: {methods}")
        for reparam in reparams:
            for method_name in methods:
                n = args.n or DEFAULT_N[method_name]
                lr = args.lr or DEFAULT_LR[method_name]
                n_epochs = args.epochs or get_n_epochs(run_mode, method_name)

                print(f"\n{'=' * 40}")
                print(f"Training {method_name} [{reparam}] (n={n}, epochs={n_epochs}, lr={lr})")
                print(f"{'=' * 40}")

                train_softmax_method(method_name, n, n_epochs, lr, data_dir,
                                     reparam=reparam)

    # Alt losses training
    if args.experiment in ("losses", "all"):
        # Exclude L_SC when x_max is None (no state constraints to penalize)
        available_losses = {k: v for k, v in LOSS_REGISTRY.items() if k != "L_SC"}
        loss_names = (list(available_losses.keys()) if "all" in args.loss
                      else args.loss)
        for name in loss_names:
            if name == "L_SC" and x_max is None:
                parser.error("L_SC requires state constraints (x_max is None)")
            if name not in LOSS_REGISTRY:
                parser.error(
                    f"Unknown loss: {name}. "
                    f"Available: {list(LOSS_REGISTRY.keys())}")

        lr_loss = args.lr or 3e-2
        print(f"\nLosses: {loss_names}")
        print(f"Balancing: "
              f"{'adaptive' if not args.no_balancing else 'fixed'}, "
              f"lambda0={args.lambda0}, detach={args.detach}")

        for reparam in reparams:
            for loss_name in loss_names:
                n = args.n or n_default
                n_epochs = args.epochs or get_n_epochs(run_mode, loss_name)

                print(f"\n{'=' * 40}")
                print(f"Training {loss_name} [{reparam}] ({n_epochs} epochs)")
                print(f"{'=' * 40}")

                train_one_loss(
                    loss_name, n, n_epochs, lr_loss, args.lambda0,
                    not args.no_balancing, data_dir, detach=args.detach,
                    reparam=reparam,
                )

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
