"""General-purpose utilities for differentiable time optimization experiments."""

import json
import math
import os
import pickle
import subprocess
from enum import Enum

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch


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
    "zoh": 500,
    "L_IV": 200,
    "L_EQ": 200,
    "L_CPC": 200,
    "L_CSS": 200,
    "L_defect": 200,
    "L_dyn": 200,
    "L_equi": 200,
    "L_FI": 200,
    "L_SC": 200,
    "L_PWLH": 200,
    "L_SSD": 200,
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


def euler_matrices(dt_k, A_t, B_t, Q_t, R_t):
    """Forward Euler discretization + time-scaled block-diagonal cost matrix.

    Returns (Ad, Bd, W) with the same interface as zoh_cost_matrices so the
    training loop can branch without structural changes.
    """
    n_s = A_t.shape[0]
    n_u = B_t.shape[1]
    Ad = torch.eye(n_s, dtype=A_t.dtype) + A_t * dt_k
    Bd = B_t * dt_k
    W = torch.zeros(n_s + n_u, n_s + n_u, dtype=A_t.dtype)
    W[:n_s, :n_s] = Q_t * dt_k
    W[n_s:, n_s:] = R_t * dt_k
    return Ad, Bd, W


# ============================================================================ #
# Parametrization
# ============================================================================ #

def theta_2_dt(theta, T, n, eps=5e-3):
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
# Alternative Loss Functions (Regularizers)
# ============================================================================ #

def loss_ssd(dts):
    """L_SSD = sum_k dt_k^2

    Sum of squared durations: minimized when all dt_k are equal (given fixed sum),
    penalizes non-uniform timestep distributions.
    """
    return torch.sum(dts ** 2)


def loss_iv(inputs, dts):
    """L_IV = sum_{k=0}^{n-2} dt_k * ||u_{k+1} - u_k||^2

    Weighted input variation: penalizes large input changes in long intervals.
    """
    u_stack = torch.stack(inputs)          # (n, n_u)
    du = torch.diff(u_stack, dim=0)        # (n-1, n_u)
    return torch.sum(dts[:-1] * torch.sum(du**2, dim=1))


def loss_iv_rate(inputs, dts):
    """L_IV_rate = sum_{k=0}^{n-2} ||u_{k+1} - u_k||^2 / dt_k

    Discrete approximation of integral of ||du/dt||^2: penalizes the squared
    input rate of change. Unlike L_IV, this *raises* cost on big jumps in long
    intervals (a step over large dt_k implies a large derivative). The QP's
    preferred response is to spread input change across multiple intervals,
    which forces the optimizer to allocate more samples around transitions.
    """
    u_stack = torch.stack(inputs)
    du = torch.diff(u_stack, dim=0)
    return torch.sum(torch.sum(du**2, dim=1) / dts[:-1])


def loss_iv_sym(inputs, dts):
    """L_IV_sym = sum_{k=0}^{n-2} (dt_k + dt_{k+1}) * ||u_{k+1} - u_k||^2

    Symmetric (two-sided) input variation: each Δu_k is weighted by BOTH
    adjacent intervals. Compared to L_IV (which weights only by dt_k), this
    blocks the failure modes where Δu hides in a single small dt:
      - Tiny-dt gaming: shrinking dt_k still leaves dt_{k+1}, so the penalty
        on a huge Δu_k cannot vanish.
      - Boundary-spike: the transition between fine and coarse regions is
        weighted by the plateau dt (the larger neighbor), making it expensive.
    """
    u_stack = torch.stack(inputs)
    du = torch.diff(u_stack, dim=0)
    dt_pair = dts[:-1] + dts[1:]
    return torch.sum(dt_pair * torch.sum(du**2, dim=1))


def loss_eq(inputs):
    """L_EQ = sum_k (w_k - w_bar)^2 where w_k = ||u_{k+1} - u_k||^2

    Input equidistribution: encourages uniform input variation across intervals.
    """
    u_stack = torch.stack(inputs)
    w = torch.sum(torch.diff(u_stack, dim=0)**2, dim=1)  # (n-1,)
    return torch.sum((w - w.mean())**2)


def loss_cpc(inputs, dts, u_max, tau=0.1, epsilon=0.05):
    """L_CPC = sum_k sigmoid((|u_k| - u_max + epsilon) / tau) * dt_k^2

    Constraint proximity concentration: shortens intervals near active constraints.
    """
    u_stack = torch.stack(inputs)                         # (n, n_u)
    u_abs = torch.abs(u_stack).max(dim=1).values          # (n,)
    phi = torch.sigmoid((u_abs - u_max + epsilon) / tau)
    return torch.sum(phi * dts**2)


def loss_css(inputs, dts, u_max, alpha=1.0, tau=0.1):
    """L_CSS = sum_k (a_{k+1} - a_k)^2 * dt_k^2

    Constraint switching sharpness: concentrates samples at constraint transitions.
    """
    u_stack = torch.stack(inputs)
    u_abs = torch.abs(u_stack).max(dim=1).values
    a = torch.tanh((u_abs / u_max - alpha) / tau)         # (n,)
    da = torch.diff(a)                                     # (n-1,)
    return torch.sum(da**2 * dts[:-1]**2)


def loss_defect(states, inputs, dts, W_list, Q, R):
    """L_defect = sum_k (z_k'W_k z_k - dt_k*(s_k'Qs_k + u_k'Ru_k))^2

    Intra-interval cost defect: penalizes mismatch between exact integrated cost
    and Riemann-sum approximation.
    """
    _dtype = states[0].dtype
    Q_t = torch.as_tensor(Q, dtype=_dtype)
    R_t = torch.as_tensor(R, dtype=_dtype)
    total = torch.tensor(0.0, dtype=_dtype)
    for k in range(len(inputs)):
        z_k = torch.cat([states[k], inputs[k]])
        vl_cost = z_k @ W_list[k] @ z_k
        ri_cost = dts[k] * (states[k] @ Q_t @ states[k] + inputs[k] @ R_t @ inputs[k])
        total = total + (vl_cost - ri_cost)**2
    return total


def loss_dyn(states, inputs, dts, A, B, Ad_list, Bd_list):
    """L_dyn = sum_k ||e_k||^2 / dt_k

    Dynamics consistency: penalizes deviation of ZOH discretization from Euler.
    e_k = (Ad_k - I - dt_k*A)x_k + (Bd_k - dt_k*B)u_k
    """
    I_ns = torch.eye(A.shape[0], dtype=A.dtype)
    total = torch.tensor(0.0, dtype=A.dtype)
    for k in range(len(inputs)):
        e_k = (Ad_list[k] - I_ns - dts[k] * A) @ states[k] + (Bd_list[k] - dts[k] * B) @ inputs[k]
        total = total + torch.sum(e_k**2) / dts[k]
    return total


def loss_equi(states, inputs, dts, A, B, Q, eps=1e-10):
    """Log-variance of midpoint prediction error.

    Equidistributed information: encourages uniform prediction error across intervals.
    """
    Q_t = torch.as_tensor(Q, dtype=A.dtype)
    I_list = []
    for k in range(len(inputs)):
        Ad_h, Bd_h = zoh_discretize(dts[k] / 2, A, B)
        x_mid = Ad_h @ states[k] + Bd_h @ inputs[k]
        x_hat = (states[k] + states[k + 1]) / 2
        diff = x_mid - x_hat
        I_k = diff @ Q_t @ diff
        I_list.append(I_k)
    I_t = torch.stack(I_list)
    log_I = torch.log(I_t + eps)
    return torch.sum((log_I - log_I.mean())**2)


def loss_fi(states, inputs, dts, A, B, Q, T, detach_target=True, eps=1e-10):
    """L_FI = D_KL(q || p) where q is velocity-based target, p = dt/T.

    Fisher information KL: drives timestep distribution toward velocity-weighted
    distribution.
    """
    Q_t = torch.as_tensor(Q, dtype=A.dtype)
    F_list = []
    for k in range(len(inputs)):
        v = A @ states[k + 1] + B @ inputs[k]
        F_k = v @ Q_t @ v
        F_list.append(F_k)
    F = torch.stack(F_list)
    inv_sqrt_F = 1.0 / torch.sqrt(F + eps)
    q = inv_sqrt_F / inv_sqrt_F.sum()
    if detach_target:
        q = q.detach()
    p = dts / T
    return torch.sum(q * torch.log(q / (p + eps)))


def loss_sc(states, inputs, dts, A, B, x_max, n_sub=5):
    """L_SC: intra-interval state-constraint penalty.

    Simulates the ZOH trajectory at n_sub sub-points within each interval
    and penalizes squared constraint violations that the QP cannot see.

    Args:
        states: list of n+1 torch tensors (error-coord states at sample points)
        inputs: list of n torch tensors (inputs per interval)
        dts: (n,) torch tensor of timestep durations
        A: (n_s, n_s) continuous-time state matrix
        B: (n_s, n_u) continuous-time input matrix
        x_max: dict {state_index: bound} — box constraints on states
        n_sub: number of sub-intervals per QP interval (default 5)
    """
    total = torch.tensor(0.0, dtype=A.dtype)
    for k in range(len(inputs)):
        dt_sub = dts[k] / n_sub
        Ad_sub, Bd_sub = zoh_discretize(dt_sub, A, B)
        x_m = states[k]
        for m in range(1, n_sub):  # skip m=0 (enforced by QP)
            x_m = Ad_sub @ x_m + Bd_sub @ inputs[k]
            for idx, bound in x_max.items():
                violation = torch.relu(x_m[idx].abs() - bound)
                total = total + violation ** 2
    return total


def loss_pwlh(states, inputs, dts, A, B, Q, R, n_sub=10):
    """L_PWLH: continuous cost under piecewise-linear input interpolation.

    Evaluates the quadratic cost by linearly interpolating the input between
    consecutive QP sample points and simulating the state at n_sub ZOH substeps
    within each interval.

    Compared to the ZOH cost (L_OCP, which holds the input constant), the PWLH
    cost captures how the input varies within intervals.  Its gradient w.r.t.
    dt_k is informative in transition regions where |u_{k+1} - u_k| is large,
    providing a signal to concentrate samples there.

    Args:
        states: list of n+1 torch tensors (error-coord states at sample points)
        inputs: list of n torch tensors (QP inputs, one per interval)
        dts: (n,) torch tensor of timestep durations
        A: (n_s, n_s) continuous-time state matrix
        B: (n_s, n_u) continuous-time input matrix
        Q: (n_s, n_s) state cost matrix
        R: (n_u, n_u) input cost matrix
        n_sub: number of ZOH substeps per interval (default 10)
    """
    _dtype = states[0].dtype
    Q_t = torch.as_tensor(Q, dtype=_dtype)
    R_t = torch.as_tensor(R, dtype=_dtype)
    total = torch.tensor(0.0, dtype=_dtype)
    n = len(inputs)
    for k in range(n):
        dt_sub = dts[k] / n_sub
        Ad_sub, Bd_sub = zoh_discretize(dt_sub, A, B)
        x_m = states[k]
        # PWLH: interpolate toward next input; hold at last interval.
        u_next = inputs[k + 1] if k + 1 < n else inputs[k]
        for m in range(n_sub):
            alpha = m / n_sub
            u_m = inputs[k] + alpha * (u_next - inputs[k])
            total = total + dt_sub * (x_m @ Q_t @ x_m + u_m @ R_t @ u_m)
            x_m = Ad_sub @ x_m + Bd_sub @ u_m
    return total


LOSS_REGISTRY = {
    "L_SSD": loss_ssd,
    "L_IV": loss_iv,
    "L_IV_rate": loss_iv_rate,
    "L_IV_sym": loss_iv_sym,
    "L_EQ": loss_eq,
    "L_CPC": loss_cpc,
    "L_CSS": loss_css,
    "L_defect": loss_defect,
    "L_dyn": loss_dyn,
    "L_equi": loss_equi,
    "L_FI": loss_fi,
    "L_SC": loss_sc,
    "L_PWLH": loss_pwlh,
}


def build_loss_kwargs(loss_name, states, inputs, dts, W_list, Ad_list, Bd_list,
                      A_t, B_t, Q_t, R_t, *, T, u_max, x_max=None):
    """Dispatch correct kwargs to each loss function.

    All system-specific values (T, u_max, x_max) are passed explicitly.
    """
    if loss_name == "L_SSD":
        return dict(dts=dts)
    elif loss_name == "L_IV":
        return dict(inputs=inputs, dts=dts)
    elif loss_name == "L_IV_rate":
        return dict(inputs=inputs, dts=dts)
    elif loss_name == "L_IV_sym":
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
# R-Adaptive Mesh Moves
# ============================================================================ #

def compute_importance(states, inputs, dts, Q, R, mode="cost_density"):
    """Per-timestep and per-pair importance scores for r-adaptive merge/split.

    Returns (eta_single, eta_pair):
        eta_single: shape (n,) — per-timestep score for split selection.
        eta_pair:   shape (n-1,) — per-adjacent-pair score for merge selection.

    Both arrays use averaging (not summing) over their constituents so that
    boundary timesteps with fewer neighbors are not down-weighted.

    Modes:
        "cost_density":
            single[k] = s_k'Q s_k + u_k'R u_k
            pair[k]   = (single[k] + single[k+1]) / 2
        "control_var":
            Let du2[k] = ||u_{k+1} - u_k||^2 for k in [0, n-2].
            single[k] = mean of available adjacent du2:
                interior:    (du2[k-1] + du2[k]) / 2
                edge k=0:    du2[0]
                edge k=n-1:  du2[n-2]
            pair[k]   = du2[k]
        "combined":
            geometric mean of cost_density and control_var, applied separately
            to single and pair arrays.
    """
    n = len(inputs)

    if mode == "cost_density":
        Q_t = torch.as_tensor(Q, dtype=dts.dtype, device=dts.device)
        R_t = torch.as_tensor(R, dtype=dts.dtype, device=dts.device)
        eta_t = torch.zeros(n, dtype=dts.dtype, device=dts.device)
        for k in range(n):
            s_k = states[k]
            u_k = inputs[k]
            eta_t[k] = s_k @ Q_t @ s_k + u_k @ R_t @ u_k
        eta_single = eta_t.detach().cpu().numpy()
        eta_pair = 0.5 * (eta_single[:-1] + eta_single[1:])
        return eta_single, eta_pair

    if mode == "control_var":
        if n < 2:
            return np.zeros(n), np.zeros(max(n - 1, 0))
        du2_t = torch.zeros(n - 1, dtype=dts.dtype, device=dts.device)
        for k in range(n - 1):
            du = inputs[k + 1] - inputs[k]
            du2_t[k] = torch.sum(du * du)
        du2 = du2_t.detach().cpu().numpy()
        eta_single = np.empty(n, dtype=du2.dtype)
        eta_single[0] = du2[0]
        eta_single[n - 1] = du2[n - 2]
        if n > 2:
            eta_single[1:-1] = 0.5 * (du2[:-1] + du2[1:])
        eta_pair = du2
        return eta_single, eta_pair

    if mode == "combined":
        cd_single, cd_pair = compute_importance(
            states, inputs, dts, Q, R, "cost_density")
        cv_single, cv_pair = compute_importance(
            states, inputs, dts, Q, R, "control_var")
        eta_single = np.sqrt(
            np.clip(cd_single, 0.0, None) * np.clip(cv_single, 0.0, None))
        eta_pair = np.sqrt(
            np.clip(cd_pair, 0.0, None) * np.clip(cv_pair, 0.0, None))
        return eta_single, eta_pair

    raise ValueError(f"Unknown importance mode: {mode}")


def select_merge_split(eta_single, eta_pair, theta_np, beta=1.0, rng=None):
    """Select an adjacent merge pair (j, j+1) and a split index i.

    With finite `beta`, sampled by softmax: merge ~ exp(-beta * eta_pair),
    split ~ exp(beta * eta_single). With `beta = np.inf` the selection is
    deterministic: j = argmin(eta_pair), i = argmax(eta_single) over indices
    outside {j, j+1}.

    Returns (j, i) or None when n < 3.
    """
    n = len(eta_single)
    if n < 3:
        return None

    eta_pair_arr = np.asarray(eta_pair)
    eta_single_arr = np.asarray(eta_single)

    if np.isposinf(beta):
        j = int(np.argmin(eta_pair_arr))
        valid_mask = np.ones(n, dtype=bool)
        valid_mask[j] = False
        valid_mask[j + 1] = False
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return None
        i = int(valid_idx[np.argmax(eta_single_arr[valid_idx])])
        return j, i

    if rng is None:
        rng = np.random.default_rng()

    pair_logits = -beta * eta_pair_arr
    pair_logits = pair_logits - pair_logits.max()
    pair_w = np.exp(pair_logits)
    pair_p = pair_w / pair_w.sum()
    j = int(rng.choice(n - 1, p=pair_p))

    valid = np.array([k for k in range(n) if k != j and k != j + 1])
    if len(valid) == 0:
        return None
    split_logits = beta * eta_single_arr[valid]
    split_logits = split_logits - split_logits.max()
    split_w = np.exp(split_logits)
    split_p = split_w / split_w.sum()
    i = int(rng.choice(valid, p=split_p))

    return j, i


def apply_merge_split(theta_np, j, i):
    """Apply a merge-split move in theta-space (softmax-mass preserving).

    Merge (j, j+1):  theta_new[j] = logsumexp(theta_j, theta_{j+1}); drop j+1.
    Split i (post-merge index): replace by two entries each = theta_i - log 2.

    The total softmax mass is preserved exactly (sum stays 1, dt on simplex).

    Args:
        theta_np: numpy array of shape (n, 1).
        j: merge index (0..n-2).
        i: split index in original numbering, must satisfy i not in {j, j+1}.

    Returns:
        new_theta: numpy array of shape (n, 1).
    """
    if i == j or i == j + 1:
        raise ValueError(
            f"split index i={i} cannot be in merge pair (j={j}, j+1={j+1})")

    n = theta_np.shape[0]
    theta_flat = theta_np.flatten().copy()
    dtype = theta_flat.dtype

    a, b = theta_flat[j], theta_flat[j + 1]
    M = max(a, b)
    theta_merged = M + np.log(np.exp(a - M) + np.exp(b - M))

    merged = np.delete(theta_flat, j + 1)
    merged[j] = theta_merged

    i_new = i - 1 if i > j + 1 else i

    split_val = merged[i_new] - np.array(np.log(2.0), dtype=dtype)

    new_flat = np.empty(n, dtype=dtype)
    new_flat[:i_new] = merged[:i_new]
    new_flat[i_new] = split_val
    new_flat[i_new + 1] = split_val
    new_flat[i_new + 2:] = merged[i_new + 1:]

    return new_flat.reshape(n, 1)


class RAdaptDriver:
    """Driver for r-adaptive merge/split moves with Metropolis acceptance.

    Encapsulates the frequency/temperature schedules, RNG, and the
    propose/apply/accept-or-reject cycle used inside training loops.
    Each epoch the training loop calls `maybe_step(...)`, which returns a
    dict of `radapt_*` fields ready to merge into a history entry.

    The driver requires an `evaluate_fn(epoch) -> dict` callable that, when
    invoked, returns at least `{"loss", "states", "inputs", "dts"}` from a
    forward pass at the current `theta`.
    """

    NULL_FIELDS = {
        "radapt_attempted": False, "radapt_accepted": None,
        "radapt_j": None, "radapt_i": None,
        "radapt_dL": None, "radapt_T": None,
    }

    def __init__(self, *, n_epochs, Q, R,
                 enable=False, every=10, freq_schedule=None,
                 importance="combined", beta=1.0,
                 temp_schedule=("exp", 1.0, 0.01),
                 warmup=20, cooldown=20, seed=None):
        self.enable = bool(enable)
        self.n_epochs = int(n_epochs)
        self.Q = Q
        self.R = R
        self.every = int(every)
        self.importance = importance
        self.beta = float(beta)
        self.warmup = int(warmup)
        self.cooldown = int(cooldown)
        self.rng = np.random.default_rng(seed)

        self._temp_at = self._build_temp_schedule(temp_schedule)
        self._freq_at = self._build_freq_schedule(freq_schedule)

    def _build_temp_schedule(self, temp_schedule):
        n_epochs = self.n_epochs
        if temp_schedule is None:
            def temp_at(epoch):
                return 0.0
            return temp_at

        kind, T_start, T_end = temp_schedule
        if kind not in ("linear", "exp"):
            raise ValueError(
                f"radapt temp_schedule kind must be 'linear' or 'exp', got '{kind}'")
        if T_start <= 0 or T_end <= 0:
            raise ValueError("radapt temperature endpoints must be positive")
        if kind == "linear":
            def temp_at(epoch):
                if n_epochs <= 1:
                    return float(T_end)
                t = epoch / (n_epochs - 1)
                return float(T_start + (T_end - T_start) * t)
        else:
            log_start = np.log(T_start)
            log_end = np.log(T_end)
            def temp_at(epoch):
                if n_epochs <= 1:
                    return float(T_end)
                t = epoch / (n_epochs - 1)
                return float(np.exp(log_start + (log_end - log_start) * t))
        return temp_at

    def _build_freq_schedule(self, freq_schedule):
        if freq_schedule is None:
            every = self.every
            def freq_at(epoch):
                return int(every)
            return freq_at

        f_start, f_end = freq_schedule
        active_start = self.warmup
        active_end = max(active_start, self.n_epochs - self.cooldown - 1)
        def freq_at(epoch):
            span = active_end - active_start
            if span <= 0:
                return max(1, int(f_end))
            t = (epoch - active_start) / span
            t = max(0.0, min(1.0, t))
            return max(1, int(round(f_start + (f_end - f_start) * t)))
        return freq_at

    def _is_due(self, epoch):
        if not self.enable:
            return False
        if not (self.warmup <= epoch < self.n_epochs - self.cooldown):
            return False
        every_now = self._freq_at(epoch)
        if every_now <= 0:
            return False
        return (epoch - self.warmup) % every_now == 0

    def maybe_step(self, epoch, theta, evaluate_fn, optim):
        """If a move is due this epoch, attempt it and update `theta` in place.

        Args:
            epoch: current epoch index (0-based).
            theta: torch.nn.Parameter holding softmax logits, shape (n, 1).
            evaluate_fn: callable mapping epoch -> dict containing at least
                {"loss", "states", "inputs", "dts"}.
            optim: torch optimizer holding `theta`. Its state is cleared on
                accept so stale momentum does not pull theta back.

        Returns:
            dict of `radapt_*` fields to merge into the history entry.
        """
        if not self._is_due(epoch):
            return dict(self.NULL_FIELDS)

        theta_snapshot = theta.detach().clone()
        with torch.no_grad():
            out_post = evaluate_fn(epoch)
        loss_before = float(out_post["loss"].item())

        eta_single, eta_pair = compute_importance(
            out_post["states"], out_post["inputs"], out_post["dts"],
            self.Q, self.R, mode=self.importance,
        )
        sel = select_merge_split(
            eta_single, eta_pair, theta.detach().cpu().numpy(),
            beta=self.beta, rng=self.rng,
        )
        if sel is None:
            return dict(self.NULL_FIELDS)

        j, i = sel
        theta_new_np = apply_merge_split(
            theta.detach().cpu().numpy(), j, i,
        )
        with torch.no_grad():
            theta.copy_(torch.tensor(theta_new_np, dtype=theta.dtype))
            out_after = evaluate_fn(epoch)
        loss_after = float(out_after["loss"].item())

        T_metro = self._temp_at(epoch)
        dL = loss_after - loss_before
        if dL <= 0:
            accept = True
        elif T_metro <= 1e-12:
            accept = False
        else:
            accept = self.rng.random() < math.exp(-dL / T_metro)

        if accept:
            optim.state.clear()
        else:
            with torch.no_grad():
                theta.copy_(theta_snapshot)

        return {
            "radapt_attempted": True,
            "radapt_accepted": bool(accept),
            "radapt_j": int(j),
            "radapt_i": int(i),
            "radapt_dL": float(dL),
            "radapt_T": float(T_metro),
        }


# ============================================================================ #
# Adaptive Gradient Balancing
# ============================================================================ #

class AdaptiveGradientBalancer:
    """Strategy 5A: balance L_ocp and L_reg gradient magnitudes via EMA.

    Two probe backward passes per step to estimate gradient norms, then
    returns lambda_hat = lambda_0 * EMA(||grad_ocp||) / EMA(||grad_reg||).
    Caller must do the final combined backward after calling step().
    """

    def __init__(self, lambda_0=0.3, ema_decay=0.99, eps=1e-8):
        self.lambda_0 = lambda_0
        self.ema_decay = ema_decay
        self.eps = eps
        self.ema_ocp = None
        self.ema_reg = None

    def step(self, theta, loss_ocp, loss_reg):
        """Probe gradients and return adaptive lambda_hat.

        Args:
            theta: the parameter tensor (theta for softmax simplex)
            loss_ocp: scalar tensor for the OCP loss
            loss_reg: scalar tensor for the regularizer loss

        Returns:
            lambda_hat: scalar weight for the regularizer
        """
        # Probe L_ocp gradient
        loss_ocp.backward(retain_graph=True)
        g_ocp = theta.grad.abs().max().item()
        theta.grad = None

        # Probe L_reg gradient
        loss_reg.backward(retain_graph=True)
        g_reg = theta.grad.abs().max().item()
        theta.grad = None

        # Update EMA
        if self.ema_ocp is None:
            self.ema_ocp, self.ema_reg = g_ocp, g_reg
        else:
            d = self.ema_decay
            self.ema_ocp = d * self.ema_ocp + (1 - d) * g_ocp
            self.ema_reg = d * self.ema_reg + (1 - d) * g_reg

        return self.lambda_0 * self.ema_ocp / (self.ema_reg + self.eps)


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


def _resolve_cmap_norm(deltas, cmap, norm):
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if isinstance(norm, str):
        if norm == "log":
            norm = LogNorm(vmin=np.min(deltas), vmax=np.max(deltas))
        elif norm == "linear":
            norm = Normalize(vmin=np.min(deltas), vmax=np.max(deltas))
        else:
            raise ValueError(f"Unknown norm {norm!r}; use 'linear' or 'log'")
    return cmap, norm


def _draw_zoh_segments(ax, times, x, deltas, cmap, norm):
    """Draw piecewise-constant ZOH segments colored by per-step deltas."""
    for i in range(len(x) - 1):
        ax.hlines(x[i], times[i], times[i + 1],
                  colors=cmap(norm(deltas[i + 1])), linewidth=2)
        ax.vlines(times[i + 1], x[i], x[i + 1],
                  colors=cmap(norm(deltas[i + 1])), linewidth=1)


def plot_colored(deltas, x, ax=None, cmap="plasma", norm="linear"):
    """Plot piecewise-constant signal with color indicating timestep duration.

    Args:
        deltas: timestep durations
        x: piecewise-constant signal values
        ax: optional matplotlib axis
        cmap: colormap name or Colormap instance (default "plasma")
        norm: "linear" / "log", or a matplotlib Normalize instance
            (default "linear")
    """
    times = np.cumsum(deltas)
    cmap, norm = _resolve_cmap_norm(deltas, cmap, norm)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    _draw_zoh_segments(ax, times, x, deltas, cmap, norm)

    ax.set_xlabel("Time")
    ax.set_ylabel("Input")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, location="bottom")
    cbar.set_label("deltas")

    return ax


_ZOOM_ACCENT_COLOR = '#404040'  # dark grey for inset spines / source rect / connectors / title


def add_zoom_inset(parent_ax, fig, *, zoom_xlim, loc, draw_fn=None,
                    time_lines=None, x_for_ylim=None, y_for_ylim=None,
                    bg_alpha=0.85, accent_color=_ZOOM_ACCENT_COLOR,
                    label="Zoom"):
    """Draw a zoom inset on `parent_ax` of x-range `zoom_xlim`.

    `draw_fn(axins)` should replot the data into the inset axes (this lets
    callers reuse whichever rendering they used on the parent — single-color
    line, multi-line, ZOH segments — so colors are preserved).

    `x_for_ylim`/`y_for_ylim`: optional arrays. When given, the inset's
    y-limits are computed from data within `zoom_xlim` (with one neighbor on
    each side and 5% padding) so reference lines on the parent (e.g., bounds
    far from the data) don't squash the zoom.
    """
    inset_bg = Rectangle(
        (loc[0], loc[1]), loc[2], loc[3],
        transform=parent_ax.transAxes,
        facecolor='white', edgecolor='none', alpha=bg_alpha, zorder=15,
    )
    parent_ax.add_patch(inset_bg)

    axins = parent_ax.inset_axes(loc)
    axins.set_zorder(20)
    axins.set_facecolor('none')

    if draw_fn is not None:
        draw_fn(axins)

    if time_lines is not None:
        for t_val in time_lines:
            if zoom_xlim[0] <= t_val <= zoom_xlim[1]:
                axins.axvline(t_val, color='gray', linestyle='-',
                              alpha=0.08, linewidth=0.5)

    axins.set_xlim(zoom_xlim)

    if x_for_ylim is not None and y_for_ylim is not None:
        x_arr = np.asarray(x_for_ylim)
        y_arr = np.asarray(y_for_ylim)
        mask = (x_arr >= zoom_xlim[0]) & (x_arr <= zoom_xlim[1])
        if mask.any():
            idx = np.where(mask)[0]
            lo = max(0, idx.min() - 1)
            hi = min(len(x_arr), idx.max() + 2)
            y_window = y_arr[lo:hi]
            y_lo = float(np.min(y_window))
            y_hi = float(np.max(y_window))
            pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.5 * max(abs(y_lo), 1.0)
            axins.set_ylim(y_lo - pad, y_hi + pad)

    for spine in axins.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(1.0)
    axins.tick_params(labelsize=8, colors=accent_color)

    if label:
        axins.set_title(label, fontsize=8, color=accent_color, pad=2)

    main_ylim = parent_ax.get_ylim()
    source_rect = Rectangle(
        (zoom_xlim[0], main_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        main_ylim[1] - main_ylim[0],
        edgecolor=accent_color, facecolor='none',
        linewidth=1.2, linestyle='--', zorder=5, clip_on=False,
    )
    parent_ax.add_patch(source_rect)

    con_top = ConnectionPatch(
        xyA=(zoom_xlim[1], main_ylim[1]), xyB=(0, 1),
        coordsA='data', coordsB='axes fraction',
        axesA=parent_ax, axesB=axins,
        color=accent_color, linewidth=1.0, linestyle='--', zorder=21,
    )
    con_bot = ConnectionPatch(
        xyA=(zoom_xlim[1], main_ylim[0]), xyB=(0, 0),
        coordsA='data', coordsB='axes fraction',
        axesA=parent_ax, axesB=axins,
        color=accent_color, linewidth=1.0, linestyle='--', zorder=21,
    )
    fig.add_artist(con_top)
    fig.add_artist(con_bot)

    return axins


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


def plot_training_res(sol, history, n, sol_method, cmap="plasma", norm="linear",
                       zoom_xlim=None, zoom_loc_state=None,
                       zoom_loc_input=None, ct_sol=None):
    """Plot training results (2x2 grid): loss, timesteps, state, colored input.

    Args:
        sol: solution tensors (list of torch tensors)
        history: list of dicts with 'loss' and 'dts' keys
        n: number of timesteps
        sol_method: 1 for aux (dts in sol), 2 for rep/zoh (dts in history)
        cmap: colormap for the colored-input subplot (default "plasma")
        norm: "linear" / "log" or a Normalize instance (default "linear")
        zoom_xlim: optional (t_lo, t_hi). When given, adds zoom insets on the
            state and input subplots (e.g., (0.0, 0.5) for stiff systems).
        zoom_loc_state: inset [x, y, w, h] for the state subplot (default
            bottom-right).
        zoom_loc_input: inset [x, y, w, h] for the input subplot (default
            top-right).
        ct_sol: optional dict with keys 'u_arr' and 'dts' giving a
            uniformly-sampled CT reference input trajectory to overlay on the
            colored-input subplot as a dashed grey line. When provided, a
            legend is added.
    """
    s_arr = np.array([s.detach().numpy().tolist() for s in sol[0:n]])
    u_arr = np.array([u.detach().numpy().tolist() for u in sol[n:2 * n]])
    d_arr = _extract_dts(sol, history, n, sol_method)

    fig, ax = plt.subplots(2, 2, figsize=(6.4, 6.4))
    ax[0, 0].plot([h['loss'] for h in history])
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].set_title("Loss Evolution")

    ax[0, 1].step(np.cumsum(d_arr), d_arr, where='pre')
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Timestep duration")
    ax[0, 1].set_title("Timesteps Evolution")

    plot_timegrid(d_arr, s_arr, ax[1, 0], ylabel="State", title="State Evolution")
    plot_colored(d_arr, u_arr, ax[1, 1], cmap=cmap, norm=norm)

    if ct_sol is not None:
        u_ct = np.asarray(ct_sol['u_arr']).flatten()
        dts_ct = np.asarray(ct_sol['dts']).flatten()
        n_ct = len(u_ct)
        times_ct = np.concatenate(([0.0], np.cumsum(dts_ct)))
        u_ct_step = np.concatenate((u_ct, u_ct[-1:]))
        ax[1, 1].step(
            times_ct, u_ct_step, where='post',
            linestyle='--', color='grey', linewidth=1.0, zorder=3,
        )
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        legend_elements = [
            Line2D([0], [0], color=cmap_obj(0.5), lw=2,
                   label=f"Non uniform - {n} steps"),
            Line2D([0], [0], color='grey', linestyle='--', lw=1.0,
                   label=f"Uniform - {n_ct} steps"),
        ]
        ax[1, 1].legend(handles=legend_elements, fontsize=7, loc='best')

    if zoom_xlim is not None:
        times_state_input = np.cumsum(d_arr)
        if zoom_loc_state is None:
            zoom_loc_state = [0.40, 0.08, 0.55, 0.55]  # bottom-right
        if zoom_loc_input is None:
            zoom_loc_input = [0.40, 0.40, 0.55, 0.55]  # top-right

        # State inset: replot multi-line state with default color cycle
        # so colors match the parent.
        add_zoom_inset(
            ax[1, 0], fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_state,
            draw_fn=lambda a: a.plot(times_state_input, s_arr),
            time_lines=times_state_input,
            x_for_ylim=times_state_input, y_for_ylim=s_arr,
        )

        # Input inset: replot ZOH segments with the same cmap/norm.
        cmap_r, norm_r = _resolve_cmap_norm(d_arr, cmap, norm)
        add_zoom_inset(
            ax[1, 1], fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_input,
            draw_fn=lambda a: _draw_zoh_segments(
                a, times_state_input, u_arr, d_arr, cmap_r, norm_r,
            ),
            time_lines=times_state_input,
            x_for_ylim=times_state_input, y_for_ylim=u_arr,
        )

    fig.set_constrained_layout(True)


# ============================================================================ #
# I/O
# ============================================================================ #

def save_training_res(out_dir, exp_name, sol, history, n, sol_method,
                      cmap="plasma", norm="linear",
                      zoom_xlim=None, zoom_loc_state=None,
                      zoom_loc_input=None):
    """Save training result plots to out_dir/exp_name/.

    Args:
        out_dir: base output directory
        exp_name: experiment name (used as subdirectory)
        sol: solution tensors
        history: training history
        n: number of timesteps
        sol_method: 1 for aux, 2 for rep/zoh
        cmap: colormap for the colored-input plot
        norm: "linear" / "log" or a Normalize instance
        zoom_xlim: optional (t_lo, t_hi). When given, adds a zoom inset to
            the saved state.pdf and input.pdf figures.
        zoom_loc_state: inset [x, y, w, h] for state.pdf (default bottom-right).
        zoom_loc_input: inset [x, y, w, h] for input.pdf (default top-right).
    """
    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    s_arr = np.array([s.detach().numpy().tolist() for s in sol[0:n]])
    u_arr = np.array([u.detach().numpy().tolist() for u in sol[n:2 * n]])
    d_arr = _extract_dts(sol, history, n, sol_method)

    if zoom_xlim is not None:
        if zoom_loc_state is None:
            zoom_loc_state = [0.40, 0.08, 0.55, 0.55]
        if zoom_loc_input is None:
            zoom_loc_input = [0.40, 0.40, 0.55, 0.55]
        times_si = np.cumsum(d_arr)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    plot_colored(d_arr, u_arr, ax, cmap=cmap, norm=norm)
    if zoom_xlim is not None:
        cmap_r, norm_r = _resolve_cmap_norm(d_arr, cmap, norm)
        add_zoom_inset(
            ax, fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_input,
            draw_fn=lambda a: _draw_zoh_segments(
                a, times_si, u_arr, d_arr, cmap_r, norm_r,
            ),
            time_lines=times_si,
            x_for_ylim=times_si, y_for_ylim=u_arr,
        )
    fig.savefig(f"{exp_dir}/input.pdf", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.step(np.cumsum(d_arr), d_arr, where='pre')
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

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    plot_timegrid(d_arr, s_arr, ax, ylabel="State")
    if zoom_xlim is not None:
        add_zoom_inset(
            ax, fig,
            zoom_xlim=zoom_xlim, loc=zoom_loc_state,
            draw_fn=lambda a: a.plot(times_si, s_arr),
            time_lines=times_si,
            x_for_ylim=times_si, y_for_ylim=s_arr,
        )
    fig.savefig(f"{exp_dir}/state.pdf", bbox_inches='tight')
    plt.close(fig)


def save_radapt_diagnostics(out_dir, exp_name, history, n,
                            color_accepted="#0072B2",
                            color_rejected="#E69F00"):
    """Save r-adapt move outcomes and accepted move participation as PDFs.

    Produces move_outcomes.pdf and accepted_move_participation.pdf inside
    out_dir/exp_name/.  Returns False (and does nothing) when the history
    contains no r-adapt attempts.
    """
    attempts = [h for h in history if h.get("radapt_attempted")]
    if not attempts:
        return False

    exp_dir = os.path.join(out_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    eps_attempt = np.array([h["epoch"] for h in attempts])
    dLs = np.array([h["radapt_dL"] for h in attempts])
    accepted = np.array([h["radapt_accepted"] for h in attempts])

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.scatter(eps_attempt[accepted], dLs[accepted],
               color=color_accepted, marker="o", label="accepted")
    ax.scatter(eps_attempt[~accepted], dLs[~accepted],
               color=color_rejected, marker="x", label="rejected")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\Delta L$")
    ax.legend(fontsize=7)
    fig.savefig(f"{exp_dir}/move_outcomes.pdf", bbox_inches='tight')
    plt.close(fig)

    js = np.array([h["radapt_j"] for h in attempts if h["radapt_accepted"]])
    is_ = np.array([h["radapt_i"] for h in attempts if h["radapt_accepted"]])
    bins = np.arange(-0.5, n + 0.5, 1)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.hist([js, is_], bins=bins, label=["merge j", "split i"], stacked=False)
    ax.set_xlabel("Interval index")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
    fig.savefig(f"{exp_dir}/accepted_move_participation.pdf",
                bbox_inches='tight')
    plt.close(fig)

    return True


def save_pickle(out_dir, name, data):
    """Save data to a pickle file."""
    with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_pickle(out_dir, name):
    """Load data from a pickle file."""
    with open(os.path.join(out_dir, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def load_losses_config(path=None):
    """Load shared losses config from JSON.

    If path is None, looks for losses_config.json next to utils.py.
    Returns the parsed dict, or None if the file does not exist.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "losses_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def resolve_loss_names(loss_arg, config=None):
    """Expand '--loss all' using the config's enabled list or LOSS_REGISTRY.

    Args:
        loss_arg: list from argparse (e.g. ["all"] or ["L_IV", "L_FI"])
        config: loaded config dict from load_losses_config(), or None

    Returns:
        list of loss names to run
    """
    if "all" in loss_arg:
        if config is not None and "enabled" in config:
            return list(config["enabled"])
        return list(LOSS_REGISTRY.keys())
    return list(loss_arg)


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


def save_dts_distribution(out_dir, name, history):
    """Save full dts distribution across all epochs as a numpy array.

    Args:
        out_dir: output directory
        name: pickle file name (without .pkl extension)
        history: list of history dicts, each containing a 'dts' key

    Returns:
        dts_all: numpy array of shape (n_epochs, n)
    """
    dts_all = np.stack([np.array(h['dts']).flatten() for h in history])
    save_pickle(out_dir, name, dts_all)
    return dts_all


def save_timesteps_video(out_dir, name, history=None, T=None, fps=15, dpi=100,
                         dts_all=None):
    """Save animation of timestep distribution evolution to mp4 (or gif fallback).

    Args:
        out_dir: output directory
        name: output file name (without extension)
        history: list of history dicts with 'dts' key (used if dts_all is None)
        T: total time horizon (inferred from dts_all if None)
        fps: frames per second
        dpi: resolution
        dts_all: numpy array of shape (n_epochs, n), alternative to history

    Returns:
        out_path: path to the saved video file
    """
    import matplotlib.animation as animation

    if dts_all is None:
        dts_all = np.stack([np.array(h['dts']).flatten() for h in history])
    if T is None:
        T = float(dts_all[0].sum())
    n_epochs, n = dts_all.shape
    starts_all = np.concatenate(
        [np.zeros((n_epochs, 1)), np.cumsum(dts_all, axis=1)[:, :-1]], axis=1
    )

    dt_min = dts_all.min() * 0.9
    dt_max = dts_all.max() * 1.1
    dt_uniform = T / n

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    (line,) = ax.plot([], [], 'b-', linewidth=1.2, drawstyle='steps-post')
    ax.set_xlim(0, T)
    ax.set_ylim(dt_min, dt_max)
    ax.set_xlabel("Time")
    ax.set_ylabel("Timestep duration")
    ax.axhline(dt_uniform, color='gray', linestyle='--', alpha=0.5, label='uniform')
    ax.legend(fontsize=8)
    title = ax.set_title("Epoch 0")
    fig.set_constrained_layout(True)

    def init():
        line.set_data([], [])
        return line, title

    def update(frame):
        line.set_data(starts_all[frame], dts_all[frame])
        title.set_text(f"Epoch {frame}")
        return line, title

    anim = animation.FuncAnimation(
        fig, update, frames=np.arange(n_epochs),
        init_func=init, blit=True, interval=1000 / fps,
    )

    out_path = os.path.join(out_dir, f"{name}.mp4")
    try:
        anim.save(out_path, fps=fps, dpi=dpi, writer='ffmpeg')
    except Exception:
        out_path = os.path.join(out_dir, f"{name}.gif")
        anim.save(out_path, fps=fps, dpi=dpi, writer='pillow')

    plt.close(fig)
    return out_path


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


# ============================================================================ #
# Shared Plotting Helpers (used by per-example plot scripts)
# ============================================================================ #

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for loading a method's pickle results."""
    key: str
    sol_pickle: str
    history_pickle: str
    internal_methods: tuple[str, ...]
    sol_method: int = 2
    uses_custom_n: bool = False


def load_method_results(data_dir, method_configs, suffix=""):
    """Load all available method results from pickle files.

    Args:
        data_dir: directory containing pickles
        method_configs: list of MethodConfig instances
        suffix: optional pickle name suffix

    Returns:
        results: dict of {method_name: {sol, history, n, internal_methods, sol_method}}
    """
    results = {}

    for cfg in method_configs:
        try:
            sol = load_pickle(data_dir, cfg.sol_pickle + suffix)
            history = load_pickle(data_dir, cfg.history_pickle + suffix)
            n_inferred = len(np.array(history[-1]['dts']).flatten())
            results[cfg.key] = {
                "sol": sol,
                "history": history,
                "n": n_inferred,
                "internal_methods": list(cfg.internal_methods),
                "sol_method": cfg.sol_method,
            }
        except (FileNotFoundError, OSError):
            pass

    return results


def load_loss_results(data_dir, loss_names=None, suffix=""):
    """Load all available loss results from pickle files.

    Args:
        loss_names: None (no losses), "all" (all available), or list of names.
        suffix: optional pickle name suffix

    Returns:
        results: dict of {loss_name: {"sol": ..., "history": ..., "n": ...}}
    """
    if loss_names is None:
        return {}
    if loss_names == "all":
        candidates = list(LOSS_REGISTRY.keys())
    else:
        candidates = loss_names
    results = {}

    for loss_name in candidates:
        try:
            sol = load_pickle(data_dir, f"sol_{loss_name}{suffix}")
            history = load_pickle(data_dir, f"history_{loss_name}{suffix}")
            n_inferred = len(np.array(history[-1]['dts']).flatten())
            results[loss_name] = {"sol": sol, "history": history, "n": n_inferred}
        except (FileNotFoundError, OSError):
            pass

    return results


def plot_method_results(name, result, results_dir, show=False,
                        cmap="plasma", norm="linear"):
    """Plot training results for a single method (2x2 grid per sub-method)."""
    sol_dict = result["sol"]
    history = result["history"]
    n_method = result["n"]
    sol_method = result.get("sol_method", 2)
    internal_methods = result["internal_methods"]

    for method in internal_methods:
        sol = sol_dict[method]
        hist_m = [h for h in history if h['method'] == method]
        if not hist_m:
            print(f"  No history for {name}/{method}, skipping")
            continue

        plot_training_res(sol, hist_m, n_method, sol_method=sol_method,
                          cmap=cmap, norm=norm)
        plt.suptitle(f"{name}: {method}")

        if results_dir:
            _key = method.replace(' ', '_')
            save_training_res(results_dir, f"{name}_{_key}", sol, hist_m,
                              n_method, sol_method, cmap=cmap, norm=norm)

        if not show:
            plt.close('all')


def plot_loss_results(loss_name, result, results_dir, show=False,
                      cmap="plasma", norm="linear",
                      zoom_xlim=None, zoom_loc_state=None,
                      zoom_loc_input=None, ct_sol=None):
    """Plot training results for a single loss (2x2 grid).

    When ``ct_sol`` is provided (dict with 'u_arr' and 'dts'), the input
    subplot also shows the dense uniform reference as a dashed grey line and
    a legend distinguishing the two trajectories.
    """
    sol = result["sol"]
    history = result["history"]
    n = result["n"]

    plot_training_res(
        sol, history, n, sol_method=2, cmap=cmap, norm=norm,
        zoom_xlim=zoom_xlim, zoom_loc_state=zoom_loc_state,
        zoom_loc_input=zoom_loc_input,
        ct_sol=ct_sol,
    )
    plt.suptitle(loss_name)

    if results_dir:
        save_training_res(results_dir, loss_name, sol, history, n, sol_method=2,
                          cmap=cmap, norm=norm,
                          zoom_xlim=zoom_xlim,
                          zoom_loc_state=zoom_loc_state,
                          zoom_loc_input=zoom_loc_input)

    if not show:
        plt.close('all')


def plot_loss_comparison(loss_results, results_dir, show=False,
                         filename="loss_comparison"):
    """Side-by-side timestep distributions for all losses."""
    if len(loss_results) <= 1:
        return

    n_losses = len(loss_results)
    fig, axes = plt.subplots(1, n_losses, figsize=(3.2 * n_losses, 3.2))
    if n_losses == 1:
        axes = [axes]

    for ax, (loss_name, result) in zip(axes, loss_results.items()):
        history = result["history"]
        dts_final = history[-1]['dts'].flatten()
        times = np.cumsum(dts_final)
        ax.plot(times, dts_final)
        ax.set_xlabel("Time")
        ax.set_ylabel("dt")
        ax.set_title(loss_name)

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, f"{filename}.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def plot_density_analysis_grid(method_solutions, T, colors, results_dir,
                               show=False):
    """Plot sampling density vs trajectory changes for all methods."""
    n_methods = len(method_solutions)
    if n_methods == 0:
        return

    n_rows = int(np.ceil(n_methods / 2))
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.5 * n_rows),
                            squeeze=False)

    for i, (key, ms) in enumerate(method_solutions.items()):
        n_m = ms['n']
        data = extract_trajectory_data(ms, n_m)
        metrics = compute_trajectory_metrics(data, n_m, T)
        plot_density_and_changes(data, metrics, key, colors,
                                axes=axs[i // 2, i % 2])

    for j in range(i + 1, n_rows * 2):
        fig.delaxes(axs[j // 2, j % 2])

    fig.set_constrained_layout(True)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "density_analysis.pdf"),
                    bbox_inches='tight')

    if not show:
        plt.close(fig)


def print_continuous_costs(method_results, loss_results, *, s0_eval, A, B, Q,
                           R, T):
    """Evaluate and print true continuous-time costs."""
    print("=== True Continuous-Time Cost Comparison ===\n")

    for name, result in method_results.items():
        sol_dict = result["sol"]
        history = result["history"]
        n_method = result["n"]

        for method in result["internal_methods"]:
            sol = sol_dict[method]
            hist_m = [h for h in history if h['method'] == method]
            if not hist_m:
                continue

            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_method + i].detach().float()
                         for i in range(n_method)]

            try:
                true_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, s0_eval, A, B, Q, R, T,
                )
            except Exception as exc:
                print(f"{name} ({method}): cost evaluation failed: {exc}")
                continue

            print(f"{name} ({method}):")
            print(f"  Training loss (final): {hist_m[-1]['loss']:.4f}")
            print(f"  True continuous cost:  {true_cost:.4f}")
            if isinstance(dts_final, np.ndarray):
                print(f"  dt range: [{np.min(dts_final):.5f}, "
                      f"{np.max(dts_final):.5f}]")
                print(f"  dt std:   {np.std(dts_final):.5f}")
            print()

    for loss_name, result in loss_results.items():
        sol = result["sol"]
        history = result["history"]
        n_loss = result["n"]

        dts_final = history[-1]['dts']
        inputs_qp = [sol[n_loss + k].detach().float() for k in range(n_loss)]

        try:
            true_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, s0_eval, A, B, Q, R, T,
            )
        except Exception as exc:
            print(f"{loss_name}: cost evaluation failed: {exc}")
            continue

        print(f"{loss_name}:")
        print(f"  Training loss (final):  {history[-1]['loss']:.4f}")
        print(f"  Loss OCP (final):       {history[-1].get('loss_ocp', 'N/A')}")
        print(f"  Loss reg (final):       {history[-1].get('loss_reg', 'N/A')}")
        print(f"  Lambda hat (final):     {history[-1].get('lambda_hat', 'N/A')}")
        print(f"  True continuous cost:   {true_cost:.4f}")
        if isinstance(dts_final, np.ndarray):
            print(f"  dt range: [{np.min(dts_final):.5f}, "
                  f"{np.max(dts_final):.5f}]")
            print(f"  dt std:   {np.std(dts_final):.5f}")
        print()


def save_summary(method_results, loss_results, results_dir, *, s0_eval, A, B,
                 Q, R, T):
    """Compute and save metrics summary JSON."""
    summary = {}

    for name, result in method_results.items():
        for method in result["internal_methods"]:
            sol = result["sol"][method]
            hist_m = [h for h in result["history"]
                      if h['method'] == method]
            if not hist_m:
                continue
            n_m = result["n"]
            dts_final = hist_m[-1]['dts']
            inputs_qp = [sol[n_m + i].detach().float() for i in range(n_m)]
            try:
                cont_cost = evaluate_continuous_cost(
                    inputs_qp, dts_final, s0_eval, A, B, Q, R, T)
            except Exception:
                cont_cost = None
            summary[f"{name}_{method}"] = {
                "continuous_cost": cont_cost,
                "final_loss": hist_m[-1]['loss'],
            }

    for loss_name, result in loss_results.items():
        sol = result["sol"]
        history = result["history"]
        n_loss = result["n"]
        dts_final = history[-1]['dts']
        inputs_qp = [sol[n_loss + k].detach().float() for k in range(n_loss)]
        try:
            cont_cost = evaluate_continuous_cost(
                inputs_qp, dts_final, s0_eval, A, B, Q, R, T)
        except Exception:
            cont_cost = None
        entry = {
            "continuous_cost": cont_cost,
            "final_loss": history[-1]['loss'],
        }
        if 'loss_ocp' in history[-1]:
            entry["final_loss_ocp"] = history[-1]['loss_ocp']
            entry["final_loss_reg"] = history[-1]['loss_reg']
            entry["final_lambda_hat"] = history[-1]['lambda_hat']
        summary[loss_name] = entry

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, metrics in summary.items():
        cost_str = (f"{metrics['continuous_cost']:.6f}"
                    if metrics['continuous_cost'] is not None else "N/A")
        print(f"\n{label}:")
        print(f"  Continuous cost:  {cost_str}")
        print(f"  Final loss:       {metrics['final_loss']:.6f}")

    return summary


# ============================================================================ #
# Structure-aware ZOH parameter packing
# ============================================================================ #


class ZOHParamSpec:
    """Structure-aware parameter declarations + packing for ZOH cvxpylayers.

    Detects block-diagonal structure of the joint A/Q sparsity graph (via
    union-find on edges where any of A_ij, A_ji, Q_ij, Q_ji is non-negligible)
    and dispatches to one of three tiers:

    - ``general``: A or Q couple all states. Uses dense legacy-shape `Lx (m, n_s)`
      and `Lu (m, n_u)` parameters, so cost is one `sum_squares(Lx @ s + Lu @ u)`
      atom per step — identical canonicalization to the legacy formulation.

    - ``fully_diagonal``: A and Q are jointly diagonal. The discrete-time
      `Ad`, `W_xx`, and Cholesky factor `L_11` are all diagonal, so each step
      stores `Ad_diag` and `L_diag` as 1-D parameters of length n_s. Per-step
      cost is two `sum_squares` atoms.

    - ``block_diagonal``: A and Q share a block-diagonal partition with at
      least one block of size > 1. Per block i: `Ad_i (b_i, b_i)` and the
      upper-triangular `L_11_i (b_i, b_i)` (lower-triangle entries are
      structurally zero from Cholesky). Per-step cost is one `sum_squares`
      per block plus one for `L_uu`.

    All tiers stay vectorized — at most O(n_blocks) atoms per step regardless
    of n_s — to keep cvxpy canonicalization fast at large state dimension.
    """

    def __init__(self, A, B, Q, R, *, atol=1e-12):
        A_np = np.asarray(A, dtype=float)
        B_np = np.asarray(B, dtype=float)
        Q_np = np.asarray(Q, dtype=float)
        R_np = np.asarray(R, dtype=float)
        n_s = A_np.shape[0]
        n_u = B_np.shape[1]
        if A_np.shape != (n_s, n_s):
            raise ValueError(f"A must be square, got {A_np.shape}")
        if Q_np.shape != (n_s, n_s):
            raise ValueError(f"Q must be (n_s, n_s), got {Q_np.shape}")
        if R_np.shape != (n_u, n_u):
            raise ValueError(f"R must be (n_u, n_u), got {R_np.shape}")

        self.n_s = n_s
        self.n_u = n_u
        self.m = n_s + n_u
        self.atol = atol

        # Sparsity-graph based block detection (joint A ∪ Q, undirected).
        A_inf = float(np.max(np.abs(A_np))) if n_s > 0 else 0.0
        Q_inf = float(np.max(np.abs(Q_np))) if n_s > 0 else 0.0
        thr_A = atol * (A_inf if A_inf > 0 else 1.0)
        thr_Q = atol * (Q_inf if Q_inf > 0 else 1.0)

        parent = list(range(n_s))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n_s):
            for j in range(i + 1, n_s):
                if (abs(A_np[i, j]) > thr_A or abs(A_np[j, i]) > thr_A
                        or abs(Q_np[i, j]) > thr_Q or abs(Q_np[j, i]) > thr_Q):
                    union(i, j)

        groups = {}
        for i in range(n_s):
            groups.setdefault(find(i), []).append(i)
        blocks = sorted((sorted(g) for g in groups.values()), key=lambda b: b[0])
        self.blocks = blocks
        self.is_block_structured = len(blocks) > 1
        self.is_fully_diagonal = (
            self.is_block_structured and all(len(b) == 1 for b in blocks)
        )
        self.block_indices_contiguous = all(
            b == list(range(b[0], b[0] + len(b))) for b in blocks
        )

        if not self.is_block_structured:
            self.tier = "general"
        elif self.is_fully_diagonal:
            self.tier = "fully_diagonal"
        else:
            self.tier = "block_diagonal"

        # Precomputed torch index buffers (CPU; safe to use across devices).
        self._block_idx_t = [torch.tensor(b, dtype=torch.long) for b in blocks]

    # ---------------------------------------------------------------- cvxpy side

    def make_step_params(self, k):
        """Build the dict of cp.Parameter objects for a single step k."""
        n_s, n_u, m = self.n_s, self.n_u, self.m
        if self.tier == "general":
            return {
                "Ad": cp.Parameter((n_s, n_s), name=f"Ad_{k}"),
                "Bd": cp.Parameter((n_s, n_u), name=f"Bd_{k}"),
                "Lx": cp.Parameter((m, n_s), name=f"Lx_{k}"),
                "Lu": cp.Parameter((m, n_u), name=f"Lu_{k}"),
            }
        if self.tier == "fully_diagonal":
            return {
                "Ad_diag": cp.Parameter(n_s, name=f"Adiag_{k}"),
                "Bd": cp.Parameter((n_s, n_u), name=f"Bd_{k}"),
                "L_diag": cp.Parameter(n_s, name=f"Ldiag_{k}"),
                "L_xu": cp.Parameter((n_s, n_u), name=f"Lxu_{k}"),
                "L_uu": cp.Parameter((n_u, n_u), name=f"Luu_{k}"),
            }
        # block_diagonal: at least one block has size > 1.
        Ad_blocks = []
        L11_blocks = []
        for bi, block in enumerate(self.blocks):
            b = len(block)
            if b == 1:
                Ad_blocks.append(cp.Parameter(1, name=f"Ad_{k}_b{bi}"))
                L11_blocks.append(cp.Parameter(1, name=f"L11_{k}_b{bi}"))
            else:
                Ad_blocks.append(cp.Parameter((b, b), name=f"Ad_{k}_b{bi}"))
                L11_blocks.append(cp.Parameter((b, b), name=f"L11_{k}_b{bi}"))
        return {
            "Ad_blocks": Ad_blocks,
            "L11_blocks": L11_blocks,
            "Bd": cp.Parameter((n_s, n_u), name=f"Bd_{k}"),
            "L_xu": cp.Parameter((n_s, n_u), name=f"Lxu_{k}"),
            "L_uu": cp.Parameter((n_u, n_u), name=f"Luu_{k}"),
        }

    def dynamics_expr(self, s_var, u_var, params):
        """cvxpy expression for `Ad @ s + Bd @ u`."""
        if self.tier == "general":
            return params["Ad"] @ s_var + params["Bd"] @ u_var
        if self.tier == "fully_diagonal":
            return cp.multiply(params["Ad_diag"], s_var) + params["Bd"] @ u_var

        # block_diagonal
        Bd_term = params["Bd"] @ u_var  # (n_s,)
        if self.block_indices_contiguous:
            segments = []
            for block, Ad_b in zip(self.blocks, params["Ad_blocks"]):
                b = len(block)
                s_seg = s_var[block[0]:block[0] + b]
                if b == 1:
                    segments.append(cp.multiply(Ad_b, s_seg))
                else:
                    segments.append(Ad_b @ s_seg)
            stacked = segments[0] if len(segments) == 1 else cp.hstack(segments)
            return stacked + Bd_term

        # Non-contiguous fallback: place each block at its global indices.
        scalars = [None] * self.n_s
        for block, Ad_b in zip(self.blocks, params["Ad_blocks"]):
            b = len(block)
            if b == 1:
                s_seg = s_var[block[0]:block[0] + 1]
                seg = cp.multiply(Ad_b, s_seg)  # (1,)
                scalars[block[0]] = seg
            else:
                s_seg = cp.hstack([s_var[i] for i in block])
                seg = Ad_b @ s_seg  # (b,)
                for li, gi in enumerate(block):
                    scalars[gi] = seg[li:li + 1]
        return cp.hstack(scalars) + Bd_term

    def cost_expr(self, s_var, u_var, params):
        """cvxpy expression for `||L^T z||^2` for one step (z = [s; u]).

        Always emits exactly one `cp.sum_squares` atom per step, matching the
        legacy formulation's per-step cone count. Splitting into multiple
        `sum_squares` would double the SOC cone count in the diffcp affine
        map and slow the LSQR backward pass.
        """
        if self.tier == "general":
            return cp.sum_squares(params["Lx"] @ s_var + params["Lu"] @ u_var)

        if self.tier == "fully_diagonal":
            state_part = (
                cp.multiply(params["L_diag"], s_var) + params["L_xu"] @ u_var
            )
            input_part = params["L_uu"] @ u_var
            return cp.sum_squares(cp.hstack([state_part, input_part]))

        # block_diagonal: assemble the full length-m vector y = L^T @ z and
        # emit one sum_squares so the cone structure matches legacy exactly.
        L_xu = params["L_xu"]
        n_s = self.n_s

        if self.block_indices_contiguous:
            block_pieces = []
            for block, L11_b in zip(self.blocks, params["L11_blocks"]):
                b = len(block)
                s_block = s_var[block[0]:block[0] + b]
                Lxu_block = L_xu[block[0]:block[0] + b, :]
                if b == 1:
                    state_term = cp.multiply(L11_b, s_block)  # (1,)
                else:
                    state_term = L11_b @ s_block  # (b,)
                block_pieces.append(state_term + Lxu_block @ u_var)
            state_part = (
                block_pieces[0] if len(block_pieces) == 1
                else cp.hstack(block_pieces)
            )
        else:
            scalars = [None] * n_s
            for block, L11_b in zip(self.blocks, params["L11_blocks"]):
                b = len(block)
                if b == 1:
                    s_block = s_var[block[0]:block[0] + 1]
                    Lxu_block = L_xu[block[0]:block[0] + 1, :]
                    seg = cp.multiply(L11_b, s_block) + Lxu_block @ u_var  # (1,)
                    scalars[block[0]] = seg
                else:
                    s_block = cp.hstack([s_var[i] for i in block])
                    Lxu_block = L_xu[block, :]
                    seg = L11_b @ s_block + Lxu_block @ u_var  # (b,)
                    for li, gi in enumerate(block):
                        scalars[gi] = seg[li:li + 1]
            state_part = cp.hstack(scalars)

        input_part = params["L_uu"] @ u_var
        return cp.sum_squares(cp.hstack([state_part, input_part]))

    def layer_parameters(self, step_params_list):
        """Flat list of cp.Parameters for `CvxpyLayer(parameters=...)`.

        Order must match `flatten_for_layer`.
        """
        if self.tier == "general":
            Ads = [p["Ad"] for p in step_params_list]
            Bds = [p["Bd"] for p in step_params_list]
            Lxs = [p["Lx"] for p in step_params_list]
            Lus = [p["Lu"] for p in step_params_list]
            return Ads + Bds + Lxs + Lus
        if self.tier == "fully_diagonal":
            Ads = [p["Ad_diag"] for p in step_params_list]
            Bds = [p["Bd"] for p in step_params_list]
            Lds = [p["L_diag"] for p in step_params_list]
            Lxus = [p["L_xu"] for p in step_params_list]
            Luus = [p["L_uu"] for p in step_params_list]
            return Ads + Bds + Lds + Lxus + Luus
        # block_diagonal
        n_blocks = len(self.blocks)
        out = []
        for bi in range(n_blocks):
            out.extend(p["Ad_blocks"][bi] for p in step_params_list)
        for bi in range(n_blocks):
            out.extend(p["L11_blocks"][bi] for p in step_params_list)
        out.extend(p["Bd"] for p in step_params_list)
        out.extend(p["L_xu"] for p in step_params_list)
        out.extend(p["L_uu"] for p in step_params_list)
        return out

    # ---------------------------------------------------------------- torch side

    def pack_step(self, Ad, Bd, W):
        """Pack torch (Ad, Bd, W) into structured torch tensors per parameter."""
        n_s = self.n_s
        L = torch.linalg.cholesky(W)
        LT = L.T

        if self.tier == "general":
            return {
                "Ad": Ad,
                "Bd": Bd,
                "Lx": LT[:, :n_s],
                "Lu": LT[:, n_s:],
            }

        L_xu = LT[:n_s, n_s:]
        L_uu = LT[n_s:, n_s:]

        if self.tier == "fully_diagonal":
            return {
                "Ad_diag": torch.diagonal(Ad, 0),
                "Bd": Bd,
                "L_diag": torch.diagonal(LT[:n_s, :n_s], 0),
                "L_xu": L_xu,
                "L_uu": L_uu,
            }

        # block_diagonal: per-block (b, b) matrices. Cholesky gives an upper-tri
        # L^T, so the strictly-lower-tri entries of each LT11 block are exactly
        # zero — safe to store as a dense (b, b) parameter.
        LT11 = LT[:n_s, :n_s]
        Ad_blocks = []
        L11_blocks = []
        for block, idx_t in zip(self.blocks, self._block_idx_t):
            b = len(block)
            Ad_b = Ad.index_select(0, idx_t).index_select(1, idx_t)
            L11_b = LT11.index_select(0, idx_t).index_select(1, idx_t)
            if b == 1:
                Ad_blocks.append(Ad_b.reshape(1))
                L11_blocks.append(L11_b.reshape(1))
            else:
                Ad_blocks.append(Ad_b)
                L11_blocks.append(L11_b)
        return {
            "Ad_blocks": Ad_blocks,
            "L11_blocks": L11_blocks,
            "Bd": Bd,
            "L_xu": L_xu,
            "L_uu": L_uu,
        }

    def flatten_for_layer(self, packed_step_list):
        """Flat tuple of torch tensors for `layer(*tensors)`.

        Order matches `layer_parameters`.
        """
        if self.tier == "general":
            Ads = [p["Ad"] for p in packed_step_list]
            Bds = [p["Bd"] for p in packed_step_list]
            Lxs = [p["Lx"] for p in packed_step_list]
            Lus = [p["Lu"] for p in packed_step_list]
            return Ads + Bds + Lxs + Lus
        if self.tier == "fully_diagonal":
            Ads = [p["Ad_diag"] for p in packed_step_list]
            Bds = [p["Bd"] for p in packed_step_list]
            Lds = [p["L_diag"] for p in packed_step_list]
            Lxus = [p["L_xu"] for p in packed_step_list]
            Luus = [p["L_uu"] for p in packed_step_list]
            return Ads + Bds + Lds + Lxus + Luus
        # block_diagonal
        n_blocks = len(self.blocks)
        out = []
        for bi in range(n_blocks):
            out.extend(p["Ad_blocks"][bi] for p in packed_step_list)
        for bi in range(n_blocks):
            out.extend(p["L11_blocks"][bi] for p in packed_step_list)
        out.extend(p["Bd"] for p in packed_step_list)
        out.extend(p["L_xu"] for p in packed_step_list)
        out.extend(p["L_uu"] for p in packed_step_list)
        return out
