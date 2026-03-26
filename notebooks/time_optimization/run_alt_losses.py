#!/usr/bin/env python
"""Alternative loss function experiments for differentiable time optimization.

Uses ZOH3 (exact ZOH + exact integrated cost) as the base QP, with alternative
loss functions as regularizers alongside L_ocp.

Usage:
    python run_alt_losses.py --mode test --loss L_IV
    python run_alt_losses.py --mode full --loss L_IV L_FI L_dyn
    python run_alt_losses.py --mode full --loss all
    python run_alt_losses.py --mode display --run-dir data/alt_losses/260325_143022_iv_fi
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pann_clqr import create_exact_zoh_cost_clqr
from utils import (
    LOSS_REGISTRY,
    AdaptiveGradientBalancer,
    RunMode,
    compute_cross_correlation,
    compute_trajectory_metrics,
    evaluate_continuous_cost,
    extract_trajectory_data,
    get_n_epochs,
    load_pickle,
    plot_colored,
    plot_cross_correlations,
    plot_density_and_changes,
    plot_timegrid,
    save_dts_distribution,
    save_pickle,
    save_timesteps_video,
    save_training_res,
    theta_2_dt,
    zoh_cost_matrices,
)

DISC_CHOICES = ("zoh", "foe")

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


def make_run_dir(base_dir, gen_name):
    """Create data/alt_losses/yymmdd_hhmmss_gen_name/."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    dirname = f"{timestamp}_{gen_name}" if gen_name else timestamp
    run_dir = os.path.join(base_dir, dirname)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir, args):
    """Dump CLI args + git hash to run_config.json."""
    config = vars(args).copy()
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    config["git_hash"] = git_hash
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def build_loss_kwargs(loss_name, states, inputs, dts, W_list, Ad_list, Bd_list,
                      A_t, B_t, Q_t, R_t, T_val, u_max_val):
    """Dispatch correct kwargs to each loss function."""
    if loss_name == "L_IV":
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

def train_one_loss(loss_name, n, n_epochs, lr, lambda0, use_balancing, run_dir, disc="zoh"):
    """ZOH3 training loop with one alternative loss as regularizer.

    Returns:
        sol: list of torch tensors (QP solution)
        history: list of dicts with training metrics
    """
    loss_dir = os.path.join(run_dir, loss_name)
    os.makedirs(loss_dir, exist_ok=True)

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
    save_pickle(loss_dir, "sol", sol)
    save_pickle(loss_dir, "history", history)

    return sol, history


# ============================================================================ #
# Analysis
# ============================================================================ #

def analyze_one_loss(loss_dir, loss_name, sol, history, n, save_video=False):
    """Per-loss analysis: plots and metrics."""
    # Save standard training result plots (reuse existing function)
    save_training_res(os.path.dirname(loss_dir), loss_name, sol, history, n, sol_method=2)

    # Save full dts distribution across all epochs
    save_dts_distribution(loss_dir, "dts_distribution", history)

    # Optionally save timesteps evolution video
    if save_video:
        save_timesteps_video(loss_dir, "timesteps_evolution", history, T)

    # Compute continuous cost for ground truth comparison
    dts_final = history[-1]['dts']
    inputs_qp = [sol[n + k].detach().float() for k in range(n)]
    cont_cost = evaluate_continuous_cost(inputs_qp, dts_final, s0, A, B, Q, R, T)

    # Build method solution dict for extract_trajectory_data
    ms = {'sol': sol, 'history': history, 'sol_method': 2}
    data = extract_trajectory_data(ms, n)
    metrics = compute_trajectory_metrics(data, n, T)

    # Density and changes plot
    try:
        from dimp.utils import get_colors
        colors = get_colors()
    except ImportError:
        colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#D55E00']

    fig_dc, ax_dc = plt.subplots(figsize=(4.8, 3.2))
    plot_density_and_changes(data, metrics, loss_name, colors, axes=ax_dc)
    fig_dc.set_constrained_layout(True)
    fig_dc.savefig(os.path.join(loss_dir, "density_and_changes.pdf"), bbox_inches='tight')
    plt.close(fig_dc)

    # Cross-correlations
    fig_cc = plot_cross_correlations(data, metrics, loss_name, colors)
    fig_cc.set_constrained_layout(True)
    fig_cc.savefig(os.path.join(loss_dir, "cross_correlations.pdf"), bbox_inches='tight')
    plt.close(fig_cc)

    # Metrics JSON
    metrics_out = {
        "continuous_cost": cont_cost,
        "final_loss": history[-1]['loss'],
        "final_loss_ocp": history[-1]['loss_ocp'],
        "final_loss_reg": history[-1]['loss_reg'],
        "final_lambda_hat": history[-1]['lambda_hat'],
    }
    with open(os.path.join(loss_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    return metrics_out


def analyze_results(run_dir, results, n, save_video=False):
    """Cross-method comparison analysis.

    Args:
        run_dir: root output directory
        results: dict of {loss_name: (sol, history)}
        n: number of timesteps
        save_video: if True, save timesteps evolution videos
    """
    summary = {}
    for loss_name, (sol, history) in results.items():
        loss_dir = os.path.join(run_dir, loss_name)
        metrics = analyze_one_loss(loss_dir, loss_name, sol, history, n, save_video=save_video)
        summary[loss_name] = metrics

    # Comparison directory
    comp_dir = os.path.join(run_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    with open(os.path.join(comp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Comparison plot: final timesteps for all losses
    if len(results) > 1:
        fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.2))
        if len(results) == 1:
            axes = [axes]
        for ax, (loss_name, (sol, history)) in zip(axes, results.items()):
            dts_final = history[-1]['dts'].flatten()
            times = np.cumsum(dts_final)
            ax.plot(times, dts_final)
            ax.set_xlabel("Time")
            ax.set_ylabel("dt")
            ax.set_title(loss_name)
        fig.set_constrained_layout(True)
        fig.savefig(os.path.join(comp_dir, "comparison.pdf"), bbox_inches='tight')
        plt.close(fig)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for loss_name, metrics in summary.items():
        print(f"\n{loss_name}:")
        print(f"  Continuous cost:  {metrics['continuous_cost']:.6f}")
        print(f"  Final loss (ocp): {metrics['final_loss_ocp']:.6f}")
        print(f"  Final loss (reg): {metrics['final_loss_reg']:.6f}")
        print(f"  Final lambda:     {metrics['final_lambda_hat']:.6f}")


# ============================================================================ #
# Display
# ============================================================================ #

def display_results(run_dir):
    """Load pickles and regenerate plots interactively."""
    config_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    n = config.get("n", 160)
    loss_names = config.get("loss", [])
    if "all" in loss_names:
        loss_names = list(LOSS_REGISTRY.keys())

    print(f"Loading results from {run_dir}")
    print(f"Losses: {loss_names}, n={n}")

    results = {}
    for loss_name in loss_names:
        loss_dir = os.path.join(run_dir, loss_name)
        if not os.path.exists(loss_dir):
            print(f"  Skipping {loss_name}: directory not found")
            continue
        sol = load_pickle(loss_dir, "sol")
        history = load_pickle(loss_dir, "history")
        results[loss_name] = (sol, history)

        # Show interactive training result plot
        from utils import plot_training_res
        plot_training_res(sol, history, n, sol_method=2)
        plt.suptitle(loss_name)

    plt.show()


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Alternative loss function experiments for differentiable time optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["display", "test", "full"],
        help="Run mode: display (load+plot), test (5 epochs), full (default epochs)",
    )
    parser.add_argument(
        "--loss", nargs="+",
        help="Loss function(s) to test, or 'all'. Required for test/full modes.",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to previous run directory (required for display mode).",
    )
    parser.add_argument("--n", type=int, default=160, help="Number of timesteps (default: 160)")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2)")
    parser.add_argument("--lambda0", type=float, default=0.3, help="Base regularizer weight (default: 0.3)")
    parser.add_argument("--no-balancing", action="store_true", help="Use fixed lambda0 instead of adaptive")
    parser.add_argument("--gen-name", default=None, help="Human-readable suffix for output dir")
    parser.add_argument(
        "--disc", choices=DISC_CHOICES, default="zoh",
        help="Discretization method: zoh (exact, default) or foe (forward Euler)",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save mp4 animation of timestep distribution evolution for each loss",
    )

    args = parser.parse_args()

    if args.mode == "display":
        if not args.run_dir:
            parser.error("--run-dir is required for display mode")
        display_results(args.run_dir)
        return

    if not args.loss:
        parser.error("--loss is required for test/full modes")

    loss_names = list(LOSS_REGISTRY.keys()) if "all" in args.loss else args.loss
    for name in loss_names:
        if name not in LOSS_REGISTRY:
            parser.error(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    run_mode = RunMode.TEST if args.mode == "test" else RunMode.FULL
    gen_name = args.gen_name or "_".join(ln.lower().replace("l_", "") for ln in loss_names)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data", "alt_losses")
    run_dir = make_run_dir(base_dir, gen_name)
    save_run_config(run_dir, args)

    print(f"Output directory: {run_dir}")
    print(f"Losses: {loss_names}")
    print(f"Mode: {args.mode}, n={args.n}, lr={args.lr}, disc={args.disc}")
    print(f"Balancing: {'adaptive' if not args.no_balancing else 'fixed'}, lambda0={args.lambda0}")
    print()

    results = {}
    for loss_name in loss_names:
        n_epochs = args.epochs or get_n_epochs(run_mode, loss_name)
        print(f"\n{'=' * 40}")
        print(f"Training {loss_name} ({n_epochs} epochs)")
        print(f"{'=' * 40}")

        sol, history = train_one_loss(
            loss_name, args.n, n_epochs, args.lr, args.lambda0,
            not args.no_balancing, run_dir, disc=args.disc,
        )
        results[loss_name] = (sol, history)

    analyze_results(run_dir, results, args.n, save_video=args.save_video)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
