"""Parametrizations of non-uniform timestep grids."""

import torch


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
