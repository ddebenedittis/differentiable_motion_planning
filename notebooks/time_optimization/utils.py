"""General-purpose utilities for differentiable time optimization experiments."""

import os
import pickle
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize


# ============================================================================ #
# Run Mode Infrastructure
# ============================================================================ #

class RunMode(Enum):
    DISPLAY = "display"  # Load saved results, only display plots
    TEST = "test"        # Quick run with few iterations
    FULL = "full"        # Full training run


DEFAULT_EPOCHS = {
    "aux": 100,
    "rep": 200,
    "hs": 400,
    "zoh": 300,
    "zoh2": 500,
    "zoh3": 500,
}

TEST_EPOCHS = 5


def get_n_epochs(run_mode, method_key, n_epochs_override=None):
    """Return number of epochs based on run mode and method."""
    if run_mode == RunMode.TEST:
        return TEST_EPOCHS
    if n_epochs_override is not None:
        return n_epochs_override
    return DEFAULT_EPOCHS[method_key]


def get_method_run_mode(run_mode, method_key, run_overrides=None):
    """Return effective RunMode for a method, checking overrides first."""
    if run_overrides and method_key in run_overrides:
        return run_overrides[method_key]
    return run_mode


def pickle_name(base_name, n, n_default=None):
    """Append _nXX suffix when n differs from default (backward compatible)."""
    if n_default is not None and n != n_default:
        return f"{base_name}_n{n}"
    return base_name


# ============================================================================ #
# Discretization
# ============================================================================ #

def zoh_discretize(dt, A, B):
    """
    Compute exact ZOH discretization (Ad, Bd) via matrix exponential.
    Fully differentiable through torch.matrix_exp.

    Args:
        dt: Scalar timestep duration (torch tensor)
        A: Continuous-time state matrix (n_s, n_s) — torch tensor
        B: Continuous-time input matrix (n_s, n_u) — torch tensor

    Returns:
        Ad: Discrete-time state matrix (n_s, n_s)
        Bd: Discrete-time input matrix (n_s, n_u)
    """
    n_s, n_u = A.shape[0], B.shape[1]
    M = torch.zeros(n_s + n_u, n_s + n_u, device=dt.device, dtype=A.dtype)
    M[:n_s, :n_s] = A * dt
    M[:n_s, n_s:] = B * dt
    E = torch.matrix_exp(M)
    return E[:n_s, :n_s], E[:n_s, n_s:]


def zoh_cost_matrices(dt_k, A_t, B_t, Q_t, R_t):
    """
    Compute exact ZOH discretization AND exact integrated quadratic cost
    via a single Van Loan block matrix exponential.

    Returns:
        Ad: (n_s, n_s)              discrete dynamics
        Bd: (n_s, n_u)              discrete input matrix
        W:  (n_s+n_u, n_s+n_u)     PSD cost matrix such that the exact
            integrated ZOH cost is z^T W z with z = [x_k; u_k]
    """
    n_s = A_t.shape[0]
    n_u = B_t.shape[1]
    n_hat = n_s + n_u

    A_hat = torch.zeros(n_hat, n_hat, dtype=A_t.dtype, device=dt_k.device)
    A_hat[:n_s, :n_s] = A_t
    A_hat[:n_s, n_s:] = B_t

    Q_hat = torch.zeros(n_hat, n_hat, dtype=A_t.dtype, device=dt_k.device)
    Q_hat[:n_s, :n_s] = Q_t

    H = torch.zeros(2 * n_hat, 2 * n_hat, dtype=A_t.dtype, device=dt_k.device)
    H[:n_hat, :n_hat] = -A_hat.T * dt_k
    H[:n_hat, n_hat:] = Q_hat * dt_k
    H[n_hat:, n_hat:] = A_hat * dt_k

    E = torch.matrix_exp(H)

    E_br = E[n_hat:, n_hat:]
    E_tr = E[:n_hat, n_hat:]

    Ad = E_br[:n_s, :n_s]
    Bd = E_br[:n_s, n_s:]

    W_Q = E_br.T @ E_tr
    W_Q = (W_Q + W_Q.T) / 2

    W = W_Q.clone()
    W[n_s:, n_s:] = W[n_s:, n_s:] + dt_k * R_t

    W = W + 1e-8 * torch.eye(n_hat, dtype=A_t.dtype, device=dt_k.device)

    return Ad, Bd, W


def Ad_Bd_from_dt(dt, A, B):
    """Wrapper around zoh_discretize that accepts numpy A, B."""
    A_t = torch.as_tensor(A, dtype=torch.float32, device=dt.device)
    B_t = torch.as_tensor(B, dtype=torch.float32, device=dt.device)
    return zoh_discretize(dt, A_t, B_t)


def LQs_LRs_from_dt(dts, Q, R):
    """Compute Cholesky cost factors scaled by sqrt(dt).

    Args:
        dts: iterable of torch scalar tensors (timestep durations)
        Q: cost matrix (numpy or torch)
        R: cost matrix (numpy or torch)

    Returns:
        LQs: list of (n_s, n_s) torch tensors
        LRs: list of (n_u, n_u) torch tensors
    """
    Q_t = torch.as_tensor(Q, dtype=torch.float32)
    R_t = torch.as_tensor(R, dtype=torch.float32)
    LQ0 = torch.linalg.cholesky(Q_t)
    LR0 = torch.linalg.cholesky(R_t)
    LQs = [torch.sqrt(dt) * LQ0 for dt in dts]
    LRs = [torch.sqrt(dt) * LR0 for dt in dts]
    return LQs, LRs


# ============================================================================ #
# Parametrization
# ============================================================================ #

def theta_2_dt(theta, T, n, eps=1e-3):
    """Softmax simplex mapping: theta -> non-uniform timesteps summing to T.

    Args:
        theta: learnable parameters, shape (n, 1) or (n,)
        T: total time horizon
        n: number of timesteps
        eps: minimum timestep duration

    Returns:
        dts: shape (n,), positive timesteps summing to T
    """
    w = torch.softmax(theta.flatten(), dim=0)
    return eps + (T - n * eps) * w


# ============================================================================ #
# Loss Functions
# ============================================================================ #

def task_loss(states, inputs, dts, Q, R, method="unscaled"):
    """Unified task loss for trajectory optimization.

    Args:
        states: list of n torch tensors (state at each step)
        inputs: list of n torch tensors (input at each step)
        dts: torch tensor of shape (n,) or list of n scalars — timestep durations
        Q: state cost matrix (numpy or torch)
        R: input cost matrix (numpy or torch)
        method: "unscaled" or "time_scaled"

    Returns:
        Scalar torch loss
    """
    Q_th = torch.as_tensor(Q, dtype=torch.float32, device=states[0].device)
    R_th = torch.as_tensor(R, dtype=torch.float32, device=states[0].device)

    if method == "unscaled":
        return sum(
            si.t() @ Q_th @ si for si in states
        ) + sum(
            ui.t() @ R_th @ ui for ui in inputs
        )
    if method == "time_scaled":
        return sum(
            dts[i] * states[i].t() @ Q_th @ states[i] for i in range(len(states))
        ) + sum(
            dts[i] * inputs[i].t() @ R_th @ inputs[i] for i in range(len(inputs))
        )

    raise ValueError(f"Unknown method {method}")


def uniform_resampling_loss(
    inputs_qp, dts_torch, s0, A, B, Q, R, T, n_res=1000, use_exact=False,
):
    """
    Evaluate LQR cost on a dense uniform time grid.

    Creates a uniform grid, interpolates inputs using ZOH from QP solution,
    simulates state forward, and computes Riemann sum approximation.
    """
    device = dts_torch.device
    dtype = dts_torch.dtype
    n = len(dts_torch)

    A = torch.as_tensor(A, dtype=dtype, device=device)
    B = torch.as_tensor(B, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    R = torch.as_tensor(R, dtype=dtype, device=device)
    s0 = torch.as_tensor(s0, dtype=dtype, device=device)

    dt_uniform = T / n_res
    t_uniform = torch.linspace(0, T - dt_uniform, n_res, device=device, dtype=dtype)

    t_cumsum = torch.cumsum(dts_torch, dim=0)

    indices = torch.searchsorted(t_cumsum.detach(), t_uniform, right=False)
    indices = torch.clamp(indices, 0, n - 1)

    u_stack = torch.stack(inputs_qp, dim=0)
    u_interp = u_stack[indices]

    s_list = []
    s_current = s0.clone()

    if use_exact:
        dt_uniform_t = torch.tensor(dt_uniform, device=device, dtype=dtype)
        Ad_uniform, Bd_uniform = zoh_discretize(dt_uniform_t, A, B)
        for j in range(n_res):
            s_list.append(s_current)
            s_current = Ad_uniform @ s_current + Bd_uniform @ u_interp[j]
    else:
        for j in range(n_res):
            s_list.append(s_current)
            s_current = s_current + dt_uniform * (A @ s_current + B @ u_interp[j])

    s_stack = torch.stack(s_list, dim=0)

    state_cost = torch.sum((s_stack @ Q) * s_stack)
    input_cost = torch.sum((u_interp @ R) * u_interp)

    return dt_uniform * (state_cost + input_cost)


def substep_loss(
    inputs_qp, dts_torch, s0, A, B, Q, R, n_sub=10, use_exact=False,
):
    """
    Compute LQR cost with substeps within each non-uniform interval.

    For each interval k of duration dt_k, creates n_sub substeps,
    applies constant input u_k, and integrates cost contribution.
    """
    device = dts_torch.device
    dtype = dts_torch.dtype
    n = len(dts_torch)

    A = torch.as_tensor(A, dtype=dtype, device=device)
    B = torch.as_tensor(B, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    R = torch.as_tensor(R, dtype=dtype, device=device)
    s0 = torch.as_tensor(s0, dtype=dtype, device=device)

    dt_subs = dts_torch / n_sub
    u_stack = torch.stack(inputs_qp, dim=0)
    u_expanded = u_stack.repeat_interleave(n_sub, dim=0)
    dt_expanded = dt_subs.repeat_interleave(n_sub)

    total_substeps = n * n_sub

    s_list = []
    s_current = s0.clone()

    if use_exact:
        for k in range(n):
            dt_sub_k = dt_subs[k]
            Ad_k, Bd_k = zoh_discretize(dt_sub_k, A, B)
            u_k = inputs_qp[k]
            for _ in range(n_sub):
                s_list.append(s_current)
                s_current = Ad_k @ s_current + Bd_k @ u_k
    else:
        for j in range(total_substeps):
            s_list.append(s_current)
            s_current = s_current + dt_expanded[j] * (A @ s_current + B @ u_expanded[j])

    s_stack = torch.stack(s_list, dim=0)

    state_cost = torch.sum((s_stack @ Q) * s_stack, dim=1)
    input_cost = torch.sum((u_expanded @ R) * u_expanded, dim=1)

    return torch.sum(dt_expanded * (state_cost + input_cost))


def evaluate_continuous_cost(inputs_qp, dts, s0, A, B, Q, R, T, n_eval=10000):
    """
    Evaluate trajectory on a very dense grid to approximate true continuous cost.
    """
    A_t = torch.as_tensor(A, dtype=torch.float32)
    B_t = torch.as_tensor(B, dtype=torch.float32)
    Q_t = torch.as_tensor(Q, dtype=torch.float32)
    R_t = torch.as_tensor(R, dtype=torch.float32)
    s0_t = torch.as_tensor(s0, dtype=torch.float32)

    if isinstance(dts, torch.Tensor):
        dts_np = dts.detach().cpu().numpy()
    else:
        dts_np = np.array(dts)

    dt_eval = T / n_eval
    t_cumsum = np.cumsum(dts_np)

    s_current = s0_t.clone()
    total_cost = 0.0

    for j in range(n_eval):
        t_j = j * dt_eval
        k = np.searchsorted(t_cumsum, t_j, side='right')
        k = min(k, len(inputs_qp) - 1)

        u_j = inputs_qp[k]
        if isinstance(u_j, torch.Tensor):
            u_j = u_j.detach()
        else:
            u_j = torch.tensor(u_j, dtype=torch.float32)

        state_cost = float(s_current @ Q_t @ s_current)
        input_cost = float(u_j @ R_t @ u_j)
        total_cost += dt_eval * (state_cost + input_cost)

        s_current = s_current + dt_eval * (A_t @ s_current + B_t @ u_j)

    return total_cost


# ============================================================================ #
# Visualization
# ============================================================================ #

def plot_timegrid(deltas, x=None, ax=None, ylabel=None, title=None):
    """Plot time grid lines and optionally overlay a trajectory."""
    times = np.cumsum(deltas.tolist())

    if ax is None:
        fig, ax = plt.subplots()
    for t in times:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.25)

    if x is not None:
        ax.plot(times, x)

    ax.set_xlabel("Time")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def plot_colored(deltas, x, ax=None):
    """Plot piecewise-constant signal with color indicating timestep duration."""
    times = np.cumsum(deltas)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=np.min(deltas), vmax=np.max(deltas))

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    for i in range(len(x) - 1):
        ax.hlines(x[i], times[i], times[i + 1],
                  colors=cmap(norm(deltas[i + 1])), linewidth=2)
        ax.vlines(times[i + 1], x[i], x[i + 1],
                  colors=cmap(norm(deltas[i + 1])), linewidth=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Input")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("deltas")

    return ax


def _extract_dts(sol, history, n, sol_method):
    """Extract timestep array from solution/history based on method type."""
    if sol_method == 1:
        return np.concatenate(
            np.array([d.detach().numpy().tolist() for d in sol[2 * n:3 * n]])
        )
    elif sol_method == 2:
        return np.array(history[-1]['dts']).flatten()
    else:
        raise ValueError(f"Unknown sol_method {sol_method}")


def plot_training_res(sol, history, n, sol_method):
    """Plot training results: loss, colored input, state/input trajectories, timesteps.

    Args:
        sol: solution tensors (list of torch tensors)
        history: list of dicts with 'loss' and 'dts' keys
        n: number of timesteps
        sol_method: 1 for aux (dts in sol), 2 for rep/zoh (dts in history)
    """
    s_arr = np.array([s.detach().numpy().tolist() for s in sol[0:n]])
    u_arr = np.array([u.detach().numpy().tolist() for u in sol[n:2 * n]])
    d_arr = _extract_dts(sol, history, n, sol_method)

    fig, ax = plt.subplots(2, 3, figsize=(9.6, 9.6))
    ax[0, 0].plot([h['loss'] for h in history])
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].set_title("Loss Evolution")

    plot_colored(d_arr, u_arr, ax[1, 0])

    plot_timegrid(d_arr, s_arr, ax[0, 1], ylabel="State", title="State Evolution")
    plot_timegrid(d_arr, u_arr, ax[1, 1], ylabel="Input", title="Input Evolution")

    ax[0, 2].plot(np.cumsum(d_arr), d_arr)
    ax[0, 2].set_xlabel("Time")
    ax[0, 2].set_ylabel("Timestep duration")
    ax[0, 2].set_title("Timesteps Evolution")

    fig.delaxes(ax[1, 2])
    fig.set_constrained_layout(True)


# ============================================================================ #
# I/O
# ============================================================================ #

def save_training_res(out_dir, exp_name, sol, history, n, sol_method):
    """Save training result plots to out_dir/exp_name/.

    Args:
        out_dir: base output directory
        exp_name: experiment name (used as subdirectory)
        sol: solution tensors
        history: training history
        n: number of timesteps
        sol_method: 1 for aux, 2 for rep/zoh
    """
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    s_arr = np.array([s.detach().numpy().tolist() for s in sol[0:n]])
    u_arr = np.array([u.detach().numpy().tolist() for u in sol[n:2 * n]])
    d_arr = _extract_dts(sol, history, n, sol_method)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    plot_colored(d_arr, u_arr, ax)
    fig.savefig(f"{exp_dir}/input.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.plot(np.cumsum(d_arr), d_arr)
    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep duration")
    fig.savefig(f"{exp_dir}/timesteps.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.plot([h['loss'] for h in history])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(f"{exp_dir}/loss.pdf", bbox_inches='tight')
    plt.close(fig)


def save_pickle(out_dir, name, data):
    """Save data to a pickle file."""
    with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_pickle(out_dir, name):
    """Load data from a pickle file."""
    with open(os.path.join(out_dir, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


# ============================================================================ #
# Analysis
# ============================================================================ #

def extract_trajectory_data(ms, n):
    """Extract states, inputs, and timesteps from a method solution."""
    sol, history, sol_method = ms['sol'], ms['history'], ms.get('sol_method', 2)

    s_arr = np.array([sol[i].detach().numpy() for i in range(n)])
    u_arr = np.array([sol[n + i].detach().numpy() for i in range(n)]).flatten()

    if sol_method == 1:
        dts = np.concatenate([sol[2 * n + i].detach().numpy() for i in range(n)]).flatten()
    else:
        dts = np.array(history[-1]['dts']).flatten()

    times = np.cumsum(dts)
    return {'s': s_arr, 'u': u_arr, 'dts': dts, 'times': times}


def compute_trajectory_metrics(data, n, T):
    """Compute sampling density and trajectory change metrics."""
    s, u, dts = data['s'], data['u'], data['dts']
    dt_uniform = T / n

    return {
        'sampling_density': (1.0 / dts) * dt_uniform,
        'abs_u': np.abs(u),
        'delta_u': np.abs(np.diff(u)),
        'norm_s': np.linalg.norm(s, axis=1),
        'delta_s': np.linalg.norm(np.diff(s, axis=0), axis=1),
    }


def compute_cross_correlation(x, y, max_lag=20):
    """Normalized cross-correlation. Positive lag: x[k] vs y[k+lag]."""
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)
    m = min(len(x), len(y))
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.array([
        np.mean(x[:m - lag] * y[lag:m]) if lag >= 0
        else np.mean(x[-lag:m] * y[:m + lag])
        for lag in lags if (m - abs(lag)) > 0
    ])
    return lags, ccf


def plot_density_and_changes(data, metrics, method_name, colors, axes=None):
    """Plot sampling density, |Delta u|, and ||Delta s|| on the same axes."""
    times = data['times']
    ax = axes if axes is not None else plt.subplots(figsize=(3.2, 2.4))[1]

    ax.plot(times, metrics['sampling_density'], label=r'Sampling density', color=colors[0])
    ax.plot(times[:-1], metrics['delta_u'], label=r'$|\Delta u|$', color=colors[1])
    ax.plot(times[:-1], metrics['delta_s'], label=r'$\|\Delta s\|_2$', color=colors[2])
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set(ylabel='Value', title=method_name)
    ax.legend(loc='upper right', fontsize=7)


def plot_cross_correlations(data, metrics, method_name, colors, max_lag=30):
    """Plot cross-correlation between sampling density and trajectory metrics."""
    sd = metrics['sampling_density']
    times = data['times']
    n = len(sd)
    sig_bound = 1.96 / np.sqrt(n)

    quantities = [
        ('delta_u', r'$|\Delta u|$', sd[:-1]),
        ('delta_s', r'$\|\Delta s\|_2$', sd[:-1]),
        ('abs_u', r'$|u|$', sd),
        ('norm_s', r'$\|s\|_2$', sd),
        ('time', r'$t$', sd),
    ]

    metrics_with_time = {**metrics, 'time': times}

    fig, axs = plt.subplots(1, len(quantities), figsize=(2.8 * len(quantities), 2.8))

    for ax, (key, label, sd_aligned) in zip(axs, quantities):
        metric_vals = metrics_with_time[key]
        if len(metric_vals) > len(sd_aligned):
            metric_vals = metric_vals[:len(sd_aligned)]
        lags, ccf = compute_cross_correlation(sd_aligned, metric_vals, max_lag=max_lag)
        ax.plot(lags, ccf, color=colors[0], marker='o', markersize=2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(sig_bound, color=colors[5], linestyle=':', alpha=0.5)
        ax.axhline(-sig_bound, color=colors[5], linestyle=':', alpha=0.5)
        ax.set(xlabel='Lag', ylabel='CCF', title=f'SD vs {label}')

        peak_idx = np.argmax(np.abs(ccf))
        peak_lag, peak_val = lags[peak_idx], ccf[peak_idx]

        y_offset = -25 if peak_val > 0 else 15
        ax.annotate(
            f'{peak_val:.2f} @ {peak_lag}',
            xy=(peak_lag, peak_val), fontsize=8,
            xytext=(0, y_offset), textcoords='offset points',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )

    fig.suptitle(f'{method_name}', fontsize=11)
    return fig
