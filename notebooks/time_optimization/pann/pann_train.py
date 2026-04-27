#!/usr/bin/env python
"""Data generation script for Pannocchia CLQR differentiable time optimization.

Trains methods (aux, rep, hs_uniform, hs_substeps, zoh) and alternative loss
functions for learning non-uniform timesteps in constrained LQR problems.

Usage:
    python pann_train.py --mode test --experiment all
    python pann_train.py --mode full --experiment methods --method rep zoh
    python pann_train.py --mode full --experiment losses --loss L_IV L_FI
    python pann_train.py --mode full --experiment methods --method zoh --n 80 --epochs 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from pann_prob import (
    create_pann_param_clqr,
    create_pann_param_clqr_2,
    create_exact_zoh_cost_clqr,
    A, B, s0, T, Q, R, u_max, n_s, n_u,
)
from utils import (
    LOSS_REGISTRY,
    AdaptiveGradientBalancer,
    RunMode,
    build_loss_kwargs,
    euler_matrices,
    get_n_epochs,
    load_losses_config,
    pickle_name,
    resolve_loss_names,
    Ad_Bd_from_dt,
    theta_2_dt,
    zoh_cost_matrices,
    task_loss,
    uniform_resampling_loss,
    substep_loss,
    save_dts_distribution,
    save_pickle,
    save_run_config,
)

ALL_METHODS = ["aux", "rep", "hs_uniform", "hs_substeps", "zoh"]

DEFAULT_N = {
    "aux": 160, "rep": 160, "hs_uniform": 160, "hs_substeps": 160,
    "zoh": 80,
}
DEFAULT_LR = {
    "aux": 5e-4, "rep": 1e-2, "hs_uniform": 1e-2, "hs_substeps": 1e-2,
    "zoh": 1e-2,
}

DISC_CHOICES = ("foe", "zoh")


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


def train_softmax_method(method_name, n, n_epochs, lr, data_dir, n_default=160):
    """Train a softmax-based method.

    Args:
        method_name: one of "rep", "hs_uniform", "hs_substeps", "zoh"
        n: number of timesteps
        n_epochs: number of training epochs
        lr: learning rate
        data_dir: directory for pickle output
        n_default: default n for pickle naming (used by zoh)

    Returns:
        sol: solution dict or tensors
        history: list of history dicts
    """
    internal_key = _INTERNAL_METHOD_KEY[method_name]

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
# Alternative Loss Training
# ============================================================================ #

def train_one_loss(loss_name, n, n_epochs, lr, lambda0, use_balancing, data_dir,
                   disc="foe"):
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs, eta_min=lr * 0.01)

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
                A_t, B_t, Q_t, R_t, T=T, u_max=u_max,
            )
            loss_reg = loss_fn(**kwargs)

            # Combine losses
            if balancer is not None:
                lambda_hat = balancer.step(theta, loss_ocp, loss_reg)
            else:
                lambda_hat = lambda0

            loss = loss_ocp + lambda_hat * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
            optim.step()
            scheduler.step()

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
        n: number of timesteps (default: 40)
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
        raise ValueError(
            f"discretization must be 'zoh' or 'euler', got '{discretization}'")

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

    n = n or 40
    dtype = torch.float64
    A_t = torch.tensor(A, dtype=dtype)
    B_t = torch.tensor(B, dtype=dtype)
    Q_t = torch.tensor(Q, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    s0_t = torch.tensor(s0, dtype=dtype)

    for name in loss_weights:
        if name not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    if discretization == "zoh":
        _, layer, _, _, _, _, _, _ = create_exact_zoh_cost_clqr(
            n, s0, n_s, n_u, u_max,
        )
    else:
        _, layer, _, _, _ = create_pann_param_clqr_2(n, s0, A, B, Q, R, u_max)

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

            if discretization == "zoh":
                sol = layer(*Ad_list, *Bd_list, *Lx_list, *Lu_list)
            else:
                sol = layer(dts_torch)

            states = [s0_t] + [sol[k] for k in range(n)]
            inputs = [sol[n + k] for k in range(n)]

            if detach in ("reg", "all"):
                states_d = [s.detach() for s in states]
                inputs_d = [u.detach() for u in inputs]
            states_ocp = states_d if detach == "all" else states
            inputs_ocp = inputs_d if detach == "all" else inputs
            states_reg = states_d if detach in ("reg", "all") else states
            inputs_reg = inputs_d if detach in ("reg", "all") else inputs

            loss_ocp = torch.tensor(0.0, dtype=dtype)
            for k in range(n):
                z_k = torch.cat([states_ocp[k], inputs_ocp[k]])
                loss_ocp = loss_ocp + z_k @ W_list[k] @ z_k

            reg_losses = {}
            loss_reg_total = torch.tensor(0.0, dtype=dtype)
            for name, w in loss_weights.items():
                kwargs = build_loss_kwargs(
                    name, states_reg, inputs_reg, dts_torch, W_list,
                    Ad_list, Bd_list, A_t, B_t, Q_t, R_t,
                    T=T, u_max=u_max,
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
        description="Data generation for Pannocchia CLQR differentiable time optimization.",
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
    parser.add_argument("--n", type=int, default=None, help="Override timestep count")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--lambda0", type=float, default=0.3,
                        help="Base regularizer weight (default: 0.3)")
    parser.add_argument("--no-balancing", action="store_true",
                        help="Use fixed lambda0 instead of adaptive")
    parser.add_argument(
        "--disc", choices=DISC_CHOICES, default="foe",
        help="Discretization method for losses: zoh or foe (forward Euler, default)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Pickle output directory (default: data/pann_clqr_dt)",
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
    data_dir = args.data_dir or os.path.join(script_dir, "data", "pann_clqr_dt")
    os.makedirs(data_dir, exist_ok=True)

    # Default n for pickle naming (matches notebook default)
    n_default = 160

    save_run_config(data_dir, args)

    print(f"Output directory: {data_dir}")
    print(f"Mode: {args.mode}")
    print()

    # Methods training
    if args.experiment in ("methods", "all"):
        methods = ALL_METHODS if "all" in args.method else args.method
        for m in methods:
            if m not in ALL_METHODS:
                parser.error(f"Unknown method: {m}. Available: {ALL_METHODS}")

        print(f"Methods: {methods}")
        for method_name in methods:
            n = args.n or DEFAULT_N[method_name]
            lr = args.lr or DEFAULT_LR[method_name]
            epoch_key = method_name
            if method_name in ("hs_uniform", "hs_substeps"):
                epoch_key = "hs"
            n_epochs = args.epochs or get_n_epochs(run_mode, epoch_key)

            print(f"\n{'=' * 40}")
            print(f"Training {method_name} (n={n}, epochs={n_epochs}, lr={lr})")
            print(f"{'=' * 40}")

            if method_name == "aux":
                train_aux(n, n_epochs, lr, data_dir)
            else:
                train_softmax_method(
                    method_name, n, n_epochs, lr, data_dir,
                    n_default=n_default,
                )

    # Alt losses training
    if args.experiment in ("losses", "all"):
        loss_names = resolve_loss_names(args.loss, losses_cfg)
        for name in loss_names:
            if name not in LOSS_REGISTRY:
                parser.error(
                    f"Unknown loss: {name}. "
                    f"Available: {list(LOSS_REGISTRY.keys())}")

        n_loss = args.n or 160
        lr_loss = args.lr or 3e-2
        per_loss_cfg = losses_cfg.get("per_loss", {}) if losses_cfg else {}
        print(f"\nLosses: {loss_names}")
        print(f"Balancing: "
              f"{'adaptive' if not args.no_balancing else 'fixed'}, "
              f"lambda0={args.lambda0}, disc={args.disc}")

        for loss_name in loss_names:
            n_epochs = args.epochs or get_n_epochs(run_mode, loss_name)
            lambda0 = per_loss_cfg.get(loss_name, {}).get("lambda0", args.lambda0)

            print(f"\n{'=' * 40}")
            print(f"Training {loss_name} ({n_epochs} epochs, lambda0={lambda0})")
            print(f"{'=' * 40}")

            train_one_loss(
                loss_name, n_loss, n_epochs, lr_loss, lambda0,
                not args.no_balancing, data_dir, disc=args.disc,
            )

    print(f"\nResults saved to: {data_dir}")


if __name__ == "__main__":
    main()
