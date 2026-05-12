"""Differentiable discretization utilities for LTI systems."""

import torch


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
