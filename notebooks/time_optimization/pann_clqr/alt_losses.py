#!/usr/bin/env python
"""Data generation script for alternative loss function experiments.

Uses ZOH3 (exact ZOH + exact integrated cost) as the base QP, with alternative
loss functions as regularizers alongside L_ocp.

Usage:
    python alt_losses.py --mode test --loss L_IV
    python alt_losses.py --mode full --loss L_IV L_FI L_dyn
    python alt_losses.py --mode full --loss all
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

from pann_clqr import create_exact_zoh_cost_clqr
from utils import (
    LOSS_REGISTRY,
    AdaptiveGradientBalancer,
    RunMode,
    get_n_epochs,
    save_dts_distribution,
    save_pickle,
    theta_2_dt,
    zoh_cost_matrices,
)

DISC_CHOICES = ("foe", "zoh")

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


# ============================================================================ #
# Helpers
# ============================================================================ #

def euler_matrices(dt_k, A_t, B_t, Q_t, R_t):
    """Forward Euler discretization + time-scaled block-diagonal cost matrix.

    Returns (Ad, Bd, W) with the same interface as zoh_cost_matrices so the
    training loop can branch between ZOH and FOE without structural changes.
    """
    n_s = A_t.shape[0]
    n_u = B_t.shape[1]
    Ad = torch.eye(n_s, dtype=A_t.dtype) + A_t * dt_k
    Bd = B_t * dt_k
    W = torch.zeros(n_s + n_u, n_s + n_u, dtype=A_t.dtype)
    W[:n_s, :n_s] = Q_t * dt_k
    W[n_s:, n_s:] = R_t * dt_k
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
                      A_t, B_t, Q_t, R_t, T_val, u_max_val):
    """Dispatch correct kwargs to each loss function."""
    if loss_name == "L_SSD":
        return dict(dts=dts)
    elif loss_name == "L_IV":
        return dict(inputs=inputs, dts=dts)
    elif loss_name == "L_EQ":
        return dict(inputs=inputs)
    elif loss_name == "L_CPC":
        return dict(inputs=inputs, dts=dts, u_max=u_max_val)
    elif loss_name == "L_CSS":
        return dict(inputs=inputs, dts=dts, u_max=u_max_val)
    elif loss_name == "L_defect":
        return dict(states=states, inputs=inputs, dts=dts, W_list=W_list, Q=Q_t, R=R_t)
    elif loss_name == "L_dyn":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t,
                    Ad_list=Ad_list, Bd_list=Bd_list)
    elif loss_name == "L_equi":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t, Q=Q_t)
    elif loss_name == "L_FI":
        return dict(states=states, inputs=inputs, dts=dts, A=A_t, B=B_t, Q=Q_t, T=T_val)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ============================================================================ #
# Training
# ============================================================================ #

def train_one_loss(loss_name, n, n_epochs, lr, lambda0, use_balancing, data_dir, disc="foe"):
    """ZOH3 training loop with one alternative loss as regularizer.

    Returns:
        sol: list of torch tensors (QP solution)
        history: list of dicts with training metrics
    """
    # Use float64 to match cvxpylayers output dtype
    dtype = torch.float64
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    s0_t = torch.tensor(s0, dtype=dtype)

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    _, layer, _, _, _, _, _, _ = create_exact_zoh_cost_clqr(n, s0, n_s, n_u, u_max)

    balancer = AdaptiveGradientBalancer(lambda_0=lambda0) if use_balancing else None
    loss_fn = LOSS_REGISTRY[loss_name]

    history = []
    sol = None

    with tqdm(total=n_epochs, desc=loss_name) as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            dts_torch = theta_2_dt(theta, T, n)

            # Compute discretization parameters (ZOH or FOE)
            compute_matrices = zoh_cost_matrices if disc == "zoh" else euler_matrices
            Ad_list, Bd_list, Lx_list, Lu_list, W_list = [], [], [], [], []
            for k in range(n):
                Ad_k, Bd_k, W_k = compute_matrices(dts_torch[k], A_t, B_t, Q_t, R_t)
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

            # L_ocp: exact integrated cost
            loss_ocp = torch.tensor(0.0, dtype=dtype)
            for k in range(n):
                z_k = torch.cat([states[k], inputs[k]])
                loss_ocp = loss_ocp + z_k @ W_list[k] @ z_k

            # L_reg: alternative loss
            kwargs = build_loss_kwargs(
                loss_name, states, inputs, dts_torch, W_list, Ad_list, Bd_list,
                A_t, B_t, Q_t, R_t, T, u_max,
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
            })

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ocp=f"{loss_ocp.item():.4f}",
                reg=f"{loss_reg.item():.4f}",
                lam=f"{lambda_hat:.4f}" if isinstance(lambda_hat, float) else f"{lambda_hat:.4f}",
            )

    # Save results
    save_pickle(data_dir, f"sol_{loss_name}", sol)
    save_pickle(data_dir, f"history_{loss_name}", history)
    save_dts_distribution(data_dir, f"dts_dist_{loss_name}", history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Data generation for alternative loss function experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["test", "full"],
        help="Run mode: test (5 epochs) or full (default epochs per loss)",
    )
    parser.add_argument(
        "--loss", nargs="+", required=True,
        help="Loss function(s) to train, or 'all'. Available: " + str(list(LOSS_REGISTRY.keys())),
    )
    parser.add_argument("--n", type=int, default=160, help="Number of timesteps (default: 160)")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate (default: 3e-2)")
    parser.add_argument("--lambda0", type=float, default=0.3, help="Base regularizer weight (default: 0.3)")
    parser.add_argument("--no-balancing", action="store_true", help="Use fixed lambda0 instead of adaptive")
    parser.add_argument(
        "--disc", choices=DISC_CHOICES, default="foe",
        help="Discretization method: zoh (exact, default) or foe (forward Euler)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle output directory (default: data/alt_losses)",
    )

    args = parser.parse_args()

    loss_names = list(LOSS_REGISTRY.keys()) if "all" in args.loss else args.loss
    for name in loss_names:
        if name not in LOSS_REGISTRY:
            parser.error(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    run_mode = RunMode.TEST if args.mode == "test" else RunMode.FULL

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "alt_losses")
    os.makedirs(data_dir, exist_ok=True)

    save_run_config(data_dir, args)

    print(f"Output directory: {data_dir}")
    print(f"Losses: {loss_names}")
    print(f"Mode: {args.mode}, n={args.n}, lr={args.lr}, disc={args.disc}")
    print(f"Balancing: {'adaptive' if not args.no_balancing else 'fixed'}, lambda0={args.lambda0}")
    print()

    for loss_name in loss_names:
        n_epochs = args.epochs or get_n_epochs(run_mode, loss_name)

        print(f"\n{'=' * 40}")
        print(f"Training {loss_name} ({n_epochs} epochs)")
        print(f"{'=' * 40}")

        train_one_loss(
            loss_name, args.n, n_epochs, args.lr, args.lambda0,
            not args.no_balancing, data_dir, disc=args.disc,
        )

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
