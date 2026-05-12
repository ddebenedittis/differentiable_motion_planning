"""Continuous-time cost evaluation for non-uniform discretizations."""

import numpy as np
import torch

from .discretization import zoh_discretize


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
