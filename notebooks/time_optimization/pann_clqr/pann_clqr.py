"""Pannocchia-specific QP builders for differentiable time optimization.

Each function creates its own local CVXPY variables and parameters (no globals)
and returns everything the caller needs, including a CvxpyLayer when applicable.
"""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np


def create_pann_clqr(n, s0, A, B, Q, R, dt, u_max=1.0):
    """Baseline uniform-timestep constrained LQR (Euler discretization).

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        A, B: system matrices
        Q, R: cost matrices
        dt: uniform timestep duration
        u_max: input constraint bound

    Returns:
        problem: cp.Problem
        s_vars: list of n+1 state variables (s_vars[0] is s0 constant)
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

    problem = cp.Problem(objective, constraints)
    return problem, s_vars, u_vars


def create_pann_param_clqr(n, s0, A, B, Q, R, s_bar, u_bar, u_max=1.0, T=10.0):
    """Aux method: linearized dynamics with delta_k auxiliary variables.

    Args:
        n: number of timesteps
        s0: initial state (numpy array, used as cp.Parameter value)
        A, B: system matrices
        Q, R: cost matrices
        s_bar: list of n linearization states (numpy arrays)
        u_bar: list of n linearization inputs (numpy arrays)
        u_max: input constraint bound
        T: total time horizon

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state expressions (s_vars[0] is s0 constant)
        u_vars: list of n input variables
        delta_vars: list of n delta variables
        dts_params: list of n dt parameters
    """
    n_s = A.shape[0]
    n_u = B.shape[1]

    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]
    delta_vars = [cp.Variable(1, name=f"deltas_{i}") for i in range(n)]
    dts_params = [cp.Parameter(1, nonneg=True, name=f"dts_{i}") for i in range(n)]

    objective = cp.Minimize(
        cp.sum([cp.quad_form(s_vars[i + 1], Q) * dts_params[i] for i in range(n)])
        + cp.sum([cp.quad_form(u_vars[i], R) * dts_params[i] for i in range(n)])
        + 10**3 * cp.sum([cp.square(delta_vars[i] - dts_params[i]) for i in range(n)])
    )

    dynamics_constraints = [
        s_vars[i + 1] == s_bar[i]
        + dts_params[i] * (A @ s_bar[i] + B @ u_bar[i])
        + (np.eye(n_s) + dts_params[i] * A) @ (s_vars[i] - s_bar[i])
        + dts_params[i] * B @ (u_vars[i] - u_bar[i])
        + (delta_vars[i] - dts_params[i]) * (A @ s_bar[i] + B @ u_bar[i])
        for i in range(n)
    ]

    input_limits = [cp.abs(u_vars[i]) <= u_max for i in range(n)]

    timestep_constraints = [
        delta_vars[i] >= 0 for i in range(n)
    ] + [
        cp.sum(delta_vars[0:n]) == T
    ]

    constraints = dynamics_constraints + input_limits + timestep_constraints
    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=dts_params,
        variables=s_vars[1:] + u_vars + delta_vars,
    )

    return problem, layer, s_vars, u_vars, delta_vars, dts_params


def create_pann_param_clqr_2(n, s0, A, B, Q, R, u_max=1.0):
    """Rep method: dt as direct CVXPY parameters (Euler dynamics).

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        A, B: system matrices
        Q, R: cost matrices
        u_max: input constraint bound

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state expressions
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

    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[dts_param],
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, dts_param


def create_exact_param_pann_clqr(n, s0, n_s, n_u, Q, R, u_max=1.0):
    """ZOH method: Ad/Bd as parameters (no cost scaling).

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        n_s: state dimension
        n_u: input dimension
        Q, R: cost matrices
        u_max: input constraint bound

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state expressions
        u_vars: list of n input variables
        Aps: list of n Ad parameter matrices
        Bps: list of n Bd parameter matrices
    """
    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]
    Aps = [cp.Parameter((n_s, n_s), name=f"Ad_{k}") for k in range(n)]
    Bps = [cp.Parameter((n_s, n_u), name=f"Bd_{k}") for k in range(n)]

    objective = cp.sum([
        cp.quad_form(s_vars[k + 1], Q) + cp.quad_form(u_vars[k], R)
        for k in range(n)
    ])

    constraints = [
        s_vars[k + 1] == Aps[k] @ s_vars[k] + Bps[k] @ u_vars[k]
        for k in range(n)
    ] + [
        cp.abs(u_vars[k]) <= u_max for k in range(n)
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=Aps + Bps,
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, Aps, Bps


def create_exact_param_pann_clqr_2(n, s0, n_s, n_u, u_max=1.0):
    """ZOH2 method: Ad/Bd + sqrt(dt)-scaled cost (LQ, LR as parameters).

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        n_s: state dimension
        n_u: input dimension
        u_max: input constraint bound

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state expressions
        u_vars: list of n input variables
        Aps: list of n Ad parameter matrices
        Bps: list of n Bd parameter matrices
        LQs: list of n Cholesky Q parameter matrices
        LRs: list of n Cholesky R parameter matrices
    """
    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]
    Aps = [cp.Parameter((n_s, n_s), name=f"Ad_{k}") for k in range(n)]
    Bps = [cp.Parameter((n_s, n_u), name=f"Bd_{k}") for k in range(n)]
    LQs = [cp.Parameter((n_s, n_s), PSD=True, name=f"Q_{k}") for k in range(n)]
    LRs = [cp.Parameter((n_u, n_u), PSD=True, name=f"R_{k}") for k in range(n)]

    objective = cp.sum([
        cp.sum_squares(LQs[k] @ s_vars[k + 1]) + cp.sum_squares(LRs[k] @ u_vars[k])
        for k in range(n)
    ])

    constraints = [
        s_vars[k + 1] == Aps[k] @ s_vars[k] + Bps[k] @ u_vars[k]
        for k in range(n)
    ] + [
        cp.abs(u_vars[k]) <= u_max for k in range(n)
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=Aps + Bps + LQs + LRs,
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, Aps, Bps, LQs, LRs


def create_exact_zoh_cost_clqr(n, s0, n_s, n_u, u_max=1.0):
    """ZOH3 method: exact ZOH dynamics + exact integrated quadratic cost.

    The Cholesky factor L of the cost matrix W is split into Lx (state columns)
    and Lu (input columns) so the expression Lx @ x_k + Lu @ u_k avoids
    cp.hstack on 1-D variables.

    Note: cost uses state at the BEGINNING of each interval (x_k, u_k).

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        n_s: state dimension
        n_u: input dimension
        u_max: input constraint bound

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state expressions
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

    problem = cp.Problem(cp.Minimize(objective), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=Aps + Bps + Lxs + Lus,
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, Aps, Bps, Lxs, Lus
