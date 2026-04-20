"""Stiff system LTI QP builders for differentiable time optimization.

The system has three states with well-separated time constants (0.1s, 10s, 100s),
making non-uniform timestep placement especially beneficial.

Each function creates its own local CVXPY variables and parameters (no globals)
and returns everything the caller needs, including a CvxpyLayer when applicable.

No error coordinates are needed since the goal state is the origin.
"""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np


def create_stiff_sys_baseline_clqr(n, s0, A, B, Q, R, dt, u_max, x_max):
    """Baseline uniform-timestep constrained LQR with state constraints.

    Args:
        n: number of timesteps
        s0: initial state (numpy array, shape (3,))
        A, B: continuous-time system matrices
        Q, R: cost matrices
        dt: uniform timestep duration
        u_max: input constraint bound (scalar)
        x_max: dict {state_index: bound} for state constraints, or None

    Returns:
        problem: cp.Problem
        s_vars: list of n+1 state variables
        u_vars: list of n input variables
    """
    n_s = A.shape[0]
    n_u = B.shape[1]

    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]

    objective = cp.Minimize(
        cp.sum([cp.quad_form(s_vars[i + 1], Q) for i in range(n)]) * dt
        + cp.sum([cp.quad_form(u_vars[i], R) for i in range(n)]) * dt
    )

    constraints = [
        s_vars[i + 1] == s_vars[i] + (A @ s_vars[i] + B @ u_vars[i]) * dt
        for i in range(n)
    ] + [
        cp.abs(u_vars[i]) <= u_max for i in range(n)
    ]

    if x_max is not None:
        for idx, bound in x_max.items():
            constraints += [
                s_vars[k + 1][idx] <= bound for k in range(n)
            ] + [
                s_vars[k + 1][idx] >= -bound for k in range(n)
            ]

    problem = cp.Problem(objective, constraints)
    return problem, s_vars, u_vars


def create_stiff_sys_rep_clqr(n, s0, A, B, Q, R, u_max, x_max):
    """Rep method: dt as direct CVXPY parameters (Euler dynamics).

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        A, B: continuous-time system matrices
        Q, R: cost matrices
        u_max: input constraint bound (scalar)
        x_max: dict {state_index: bound} for state constraints, or None

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state variables
        u_vars: list of n input variables
        dts_param: single cp.Parameter of shape (n,)
    """
    n_s = A.shape[0]
    n_u = B.shape[1]

    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]
    dts_param = cp.Parameter(n, nonneg=True, name='dts')

    objective = cp.Minimize(
        cp.sum([cp.quad_form(s_vars[i + 1], Q) * dts_param[i] for i in range(n)])
        + cp.sum([cp.quad_form(u_vars[i], R) * dts_param[i] for i in range(n)])
    )

    constraints = [
        s_vars[i + 1] == s_vars[i] + dts_param[i] * (A @ s_vars[i] + B @ u_vars[i])
        for i in range(n)
    ] + [
        cp.abs(u_vars[i]) <= u_max for i in range(n)
    ]

    if x_max is not None:
        for idx, bound in x_max.items():
            constraints += [
                s_vars[k + 1][idx] <= bound for k in range(n)
            ] + [
                s_vars[k + 1][idx] >= -bound for k in range(n)
            ]

    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[dts_param],
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, dts_param


def create_stiff_sys_zoh_clqr(n, s0, n_s, n_u, u_max, x_max):
    """ZOH3 method: exact ZOH dynamics + exact integrated quadratic cost.

    The Cholesky factor L of the cost matrix W is split into Lx (state columns)
    and Lu (input columns) so the expression Lx @ x_k + Lu @ u_k avoids
    cp.hstack on 1-D variables.

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        n_s: state dimension
        n_u: input dimension
        u_max: input constraint bound (scalar)
        x_max: dict {state_index: bound} for state constraints, or None

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state variables
        u_vars: list of n input variables
        Aps: list of n Ad parameter matrices
        Bps: list of n Bd parameter matrices
        Lxs: list of n Lx parameter matrices
        Lus: list of n Lu parameter matrices
    """
    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]
    Aps = [cp.Parameter((n_s, n_s), name=f"Ad3_{k}") for k in range(n)]
    Bps = [cp.Parameter((n_s, n_u), name=f"Bd3_{k}") for k in range(n)]
    Lxs = [cp.Parameter((n_s + n_u, n_s), name=f"Lx3_{k}") for k in range(n)]
    Lus = [cp.Parameter((n_s + n_u, n_u), name=f"Lu3_{k}") for k in range(n)]

    objective = cp.sum([
        cp.sum_squares(Lxs[k] @ s_vars[k] + Lus[k] @ u_vars[k])
        for k in range(n)
    ])

    constraints = [
        s_vars[k + 1] == Aps[k] @ s_vars[k] + Bps[k] @ u_vars[k]
        for k in range(n)
    ] + [
        cp.abs(u_vars[k]) <= u_max for k in range(n)
    ]

    if x_max is not None:
        for idx, bound in x_max.items():
            constraints += [
                s_vars[k + 1][idx] <= bound for k in range(n)
            ] + [
                s_vars[k + 1][idx] >= -bound for k in range(n)
            ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=Aps + Bps + Lxs + Lus,
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, Aps, Bps, Lxs, Lus
