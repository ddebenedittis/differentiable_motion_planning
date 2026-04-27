#!/usr/bin/env python
"""Data generation script for inverted pendulum differentiable time optimization.

Trains methods (rep, zoh) and alternative loss functions for learning
non-uniform timesteps in the linearized cart-pole go-to-goal problem.

All QPs are formulated in error coordinates (e = s - s_goal). Since A @ s_goal = 0
for the linearized cart-pole, both Euler and ZOH dynamics are identical in error
coordinates. The saved solutions contain error states; add s_goal to recover
actual states.

Usage:
    python invpend_train.py --mode test --experiment all
    python invpend_train.py --mode full --experiment methods --method rep
    python invpend_train.py --mode full --experiment losses --loss L_IV L_FI
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from invpend_prob import (
    create_invpend_rep_clqr,
    create_invpend_zoh_clqr,
    A, B, s0, s_goal, T, n_default, Q, R, u_max, v_max, theta_max, x_max, n_s, n_u, e0,
)
from utils import (
    LOSS_REGISTRY,
    AdaptiveGradientBalancer,
    RunMode,
    build_loss_kwargs,
    euler_matrices,
    get_n_epochs,
    load_losses_config,
    resolve_loss_names,
    save_dts_distribution,
    save_pickle,
    save_run_config,
    task_loss,
    theta_2_dt,
    zoh_cost_matrices,
)

ALL_METHODS = ["rep", "zoh"]

DEFAULT_N = {"rep": 40, "zoh": 20}
DEFAULT_LR = {"rep": 1e-2, "zoh": 1e-2}


# ============================================================================ #
# Methods Training (rep, zoh) — Softmax Parametrization
# ============================================================================ #

def _create_layer(method_name, n):
    """Create the CvxpyLayer for a given method."""
    if method_name == "rep":
        _, layer, _, _, _ = create_invpend_rep_clqr(
            n, s0, A, B, Q, R, u_max, x_max, s_goal,
        )
        return layer
    elif method_name == "zoh":
        _, layer, _, _, _, _, _, _ = create_invpend_zoh_clqr(
            n, s0, n_s, n_u, u_max, x_max, s_goal,
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


def _compute_loss(method_name, sol, dts_torch, n, W_list, e0_t):
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
            s_k = e0_t if k == 0 else sol[k - 1].to(dtype)
            u_k = sol[n + k].to(dtype)
            z_k = torch.cat([s_k, u_k])
            loss = loss + z_k @ W_list[k] @ z_k
        return loss

    raise ValueError(f"Unknown method: {method_name}")


_INTERNAL_METHOD_KEY = {
    "rep": "time_scaled",
    "zoh": "exact_zoh_integrated",
}


def train_softmax_method(method_name, n, n_epochs, lr, data_dir):
    """Train a softmax-based method.

    Args:
        method_name: one of "rep", "zoh"
        n: number of timesteps
        n_epochs: number of training epochs
        lr: learning rate
        data_dir: directory for pickle output

    Returns:
        sol: solution dict
        history: list of history dicts
    """
    internal_key = _INTERNAL_METHOD_KEY[method_name]

    dtype = torch.float32
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    e0_t = torch.tensor(e0, dtype=dtype)

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
                method_name, sol_raw, dts_torch, n, W_list, e0_t,
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
    sol_name = f"sol_{method_name}"
    hist_name = f"history_{method_name}"
    dist_name = f"dts_dist_{method_name}"

    save_pickle(data_dir, sol_name, sol)
    save_pickle(data_dir, hist_name, history)
    save_dts_distribution(data_dir, dist_name, history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Alt Losses Training (ZOH3 base + regularizer)
# ============================================================================ #

def train_one_loss(loss_name, n, n_epochs, lr, lambda0, use_balancing, data_dir,
                   detach="none"):
    """ZOH3 training loop with one alternative loss as regularizer.

    Args:
        detach: Gradient detach mode for the QP solution.
            "none"  — full gradient through cvxpylayers (default, current behavior)
            "reg"   — detach states/inputs for L_reg only (clean direct gradient)
            "all"   — detach for both L_ocp and L_reg (no cvxpylayers backward)

    Returns:
        sol: list of torch tensors (QP solution)
        history: list of dicts with training metrics
    """
    dtype = torch.float64
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    e0_t = torch.tensor(e0, dtype=dtype)

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    _, layer, _, _, _, _, _, _ = create_invpend_zoh_clqr(
        n, s0, n_s, n_u, u_max, x_max, s_goal,
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

            # Extract error states and inputs
            states = [e0_t] + [sol[k] for k in range(n)]
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
                Bd_list, A_t, B_t, Q_t, R_t, T=T, u_max=u_max, x_max=x_max,
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
    save_pickle(data_dir, f"sol_{loss_name}", sol)
    save_pickle(data_dir, f"history_{loss_name}", history)
    save_dts_distribution(data_dir, f"dts_dist_{loss_name}", history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Custom Composite Loss Training
# ============================================================================ #

def train_custom_loss(loss_weights, n=None, n_epochs=200, lr=3e-2, detach="none",
                      discretization="zoh", ocp_weight=1.0,
                      tau_schedule=None):
    """Training loop with a custom composite loss.

    The total loss is:  w_ocp * L_ocp + sum_i (w_i * L_i)

    Args:
        loss_weights: dict of {loss_name: weight}, e.g. {"L_IV": 0.2, "L_EQ": 0.3}.
                      Weights are fixed (no adaptive balancing).
        n: number of timesteps (default: n_default)
        n_epochs: number of training epochs
        lr: learning rate
        detach: gradient detach mode ("none", "reg", "all")
        discretization: "zoh" (exact ZOH via matrix exp) or "euler" (forward Euler)
        ocp_weight: weight for L_ocp (default: 1.0)
        tau_schedule: temperature schedule for the softmax. Either None (constant
                      tau=1, equivalent to no annealing) or a tuple
                      (kind, tau_start, tau_end) with kind in {"linear", "exp"}.
                      The softmax becomes  softmax(theta / tau(epoch)). Small tau
                      sharpens, large tau flattens; with Adam, only a *changing*
                      tau perturbs the optimizer (constant tau is absorbed by the
                      EMA).

    Returns:
        sol: list of torch tensors (QP solution)
        history: list of dicts with training metrics
        n: number of timesteps used
    """
    if discretization not in ("zoh", "euler"):
        raise ValueError(f"discretization must be 'zoh' or 'euler', got '{discretization}'")

    if tau_schedule is None:
        def tau_at(epoch):
            return 1.0
    else:
        kind, tau_start, tau_end = tau_schedule
        if kind not in ("linear", "exp"):
            raise ValueError(f"tau_schedule kind must be 'linear' or 'exp', got '{kind}'")
        if tau_start <= 0 or tau_end <= 0:
            raise ValueError("tau_start and tau_end must be positive")
        if kind == "linear":
            def tau_at(epoch):
                if n_epochs <= 1:
                    return float(tau_end)
                t = epoch / (n_epochs - 1)
                return float(tau_start + (tau_end - tau_start) * t)
        else:
            log_start = np.log(tau_start)
            log_end = np.log(tau_end)
            def tau_at(epoch):
                if n_epochs <= 1:
                    return float(tau_end)
                t = epoch / (n_epochs - 1)
                return float(np.exp(log_start + (log_end - log_start) * t))

    n = n or n_default
    dtype = torch.float64
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    e0_t = torch.tensor(e0, dtype=dtype)

    # Validate loss names
    for name in loss_weights:
        if name not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    if discretization == "zoh":
        _, layer, _, _, _, _, _, _ = create_invpend_zoh_clqr(
            n, s0, n_s, n_u, u_max, x_max, s_goal,
        )
    else:
        _, layer, _, _, _ = create_invpend_rep_clqr(
            n, s0, A, B, Q, R, u_max, x_max, s_goal,
        )

    disc_fn = zoh_cost_matrices if discretization == "zoh" else euler_matrices

    loss_fns = {name: LOSS_REGISTRY[name] for name in loss_weights}

    history = []
    sol = None

    label = " + ".join(f"{w}*{name}" for name, w in loss_weights.items())
    disc_tag = discretization.upper()
    with tqdm(total=n_epochs, desc=f"custom/{disc_tag} ({label})") as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            tau = tau_at(epoch)
            theta_eff = theta if tau == 1.0 else theta / tau
            dts_torch = theta_2_dt(theta_eff, T, n)

            # Compute discretization parameters
            Ad_list, Bd_list, Lx_list, Lu_list, W_list = [], [], [], [], []
            for k in range(n):
                Ad_k, Bd_k, W_k = disc_fn(
                    dts_torch[k], A_t, B_t, Q_t, R_t,
                )
                Ad_list.append(Ad_k)
                Bd_list.append(Bd_k)
                W_list.append(W_k)

                if discretization == "zoh":
                    L_k = torch.linalg.cholesky(W_k)
                    LT_k = L_k.T
                    Lx_list.append(LT_k[:, :n_s])
                    Lu_list.append(LT_k[:, n_s:])

            # Solve QP
            if discretization == "zoh":
                sol = layer(*Ad_list, *Bd_list, *Lx_list, *Lu_list)
            else:
                sol = layer(dts_torch)

            # Extract error states and inputs
            states = [e0_t] + [sol[k] for k in range(n)]
            inputs = [sol[n + k] for k in range(n)]

            # Detach if requested
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

            # Regularizer losses
            reg_losses = {}
            loss_reg_total = torch.tensor(0.0, dtype=dtype)
            for name, w in loss_weights.items():
                kwargs = build_loss_kwargs(
                    name, states_reg, inputs_reg, dts_torch, W_list,
                    Ad_list, Bd_list, A_t, B_t, Q_t, R_t,
                    T=T, u_max=u_max, x_max=x_max,
                )
                l_reg = loss_fns[name](**kwargs)
                reg_losses[name] = float(l_reg.item())
                loss_reg_total = loss_reg_total + w * l_reg

            loss = ocp_weight * loss_ocp + loss_reg_total
            loss.backward()
            optim.step()

            entry = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_ocp": float(loss_ocp.item()),
                "loss_reg_total": float(loss_reg_total.item()),
                "dts": dts_torch.detach().cpu().numpy(),
                "tau": tau,
                "detach": detach,
            }
            entry.update({f"loss_{name}": v for name, v in reg_losses.items()})
            history.append(entry)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ocp=f"{loss_ocp.item():.4f}",
                reg=f"{loss_reg_total.item():.4f}",
                tau=f"{tau:.3f}",
            )

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history, n


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Data generation for inverted pendulum differentiable time optimization.",
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
        "--loss", nargs="+", default=["default"],
        help="Loss function(s) to train. Use 'default' (the config's "
             "enabled list) or 'all' (every loss in LOSS_REGISTRY). "
             "Available: " + str(list(LOSS_REGISTRY.keys())),
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
        help="Pickle output directory (default: data/invpend_dt)",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to losses_config.json (default: time_optimization/losses_config.json). "
             "Controls which losses run when --loss default is used.",
    )

    args = parser.parse_args()

    run_mode = RunMode.TEST if args.mode == "test" else RunMode.FULL
    losses_cfg = load_losses_config(args.config)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(script_dir, "data", "invpend_dt")
    os.makedirs(data_dir, exist_ok=True)

    save_run_config(data_dir, args)

    print(f"Output directory: {data_dir}")
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
        for method_name in methods:
            n = args.n or DEFAULT_N[method_name]
            lr = args.lr or DEFAULT_LR[method_name]
            n_epochs = args.epochs or get_n_epochs(run_mode, method_name)

            print(f"\n{'=' * 40}")
            print(f"Training {method_name} (n={n}, epochs={n_epochs}, lr={lr})")
            print(f"{'=' * 40}")

            train_softmax_method(method_name, n, n_epochs, lr, data_dir)

    # Alt losses training
    if args.experiment in ("losses", "all"):
        loss_names = resolve_loss_names(args.loss, losses_cfg)
        for name in loss_names:
            if name not in LOSS_REGISTRY:
                parser.error(
                    f"Unknown loss: {name}. "
                    f"Available: {list(LOSS_REGISTRY.keys())}")

        lr_loss = args.lr or 3e-2
        per_loss_cfg = losses_cfg.get("per_loss", {}) if losses_cfg else {}
        print(f"\nLosses: {loss_names}")
        print(f"Balancing: "
              f"{'adaptive' if not args.no_balancing else 'fixed'}, "
              f"lambda0={args.lambda0}, detach={args.detach}")

        for loss_name in loss_names:
            n = args.n or n_default
            n_epochs = args.epochs or get_n_epochs(run_mode, loss_name)
            lambda0 = per_loss_cfg.get(loss_name, {}).get("lambda0", args.lambda0)

            print(f"\n{'=' * 40}")
            print(f"Training {loss_name} ({n_epochs} epochs, lambda0={lambda0})")
            print(f"{'=' * 40}")

            train_one_loss(
                loss_name, n, n_epochs, lr_loss, lambda0,
                not args.no_balancing, data_dir, detach=args.detach,
            )

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
