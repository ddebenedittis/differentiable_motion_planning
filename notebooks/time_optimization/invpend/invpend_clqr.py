"""Inverted-pendulum QP builders for differentiable time optimization.

All QP builders work in error coordinates: e = s - s_goal, where s_goal is the
terminal goal state. This is exact because A @ s_goal = 0 for the linearized
cart-pole (position does not appear in the dynamic equations). The initial state
is transformed internally as e0 = s0 - s_goal.

Each function creates its own local CVXPY variables and parameters (no globals)
and returns everything the caller needs, including a CvxpyLayer when applicable.

The returned state variables (s_vars) are in error coordinates. To recover
actual states: x_k = s_vars[k] + s_goal.
"""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

# ============================================================================ #
# System Constants (Linearized Cart-Pole)
# ============================================================================ #

m_p, M_c, l_p, g_val = 0.2, 1.0, 0.5, 9.81

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m_p * g_val / M_c, 0],
    [0, 0, 0, 1],
    [0, 0, (M_c + m_p) * g_val / (M_c * l_p), 0],
])
B = np.array([[0], [1 / M_c], [0], [-1 / (M_c * l_p)]])

s0 = np.array([0.0, 0.0, 0.1, 0.0])
s_goal = np.array([5.0, 0.0, 0.0, 0.0])
T = 5.0
n_default = 40
Q = np.diag([1.0, 0.1, 10.0, 0.1])
R = 0.01 * np.eye(1)
u_max = 10.0
v_max = 2.0
theta_max = 0.15
x_max = {1: v_max, 2: theta_max}
n_s = 4
n_u = 1
e0 = s0 - s_goal


def _validate_error_coords(s_goal, x_max):
    """Assert that s_goal is zero at all constrained state indices."""
    if x_max is not None:
        for idx in x_max:
            assert s_goal[idx] == 0.0, (
                f"State constraint at index {idx} requires s_goal[{idx}]=0 "
                f"for error-coordinate formulation, got {s_goal[idx]}"
            )


def create_invpend_baseline_clqr(n, s0, A, B, Q, R, dt, u_max, x_max, s_goal):
    """Baseline uniform-timestep constrained LQR with state + terminal constraints.

    Args:
        n: number of timesteps
        s0: initial state (numpy array, shape (4,))
        A, B: continuous-time system matrices
        Q, R: cost matrices
        dt: uniform timestep duration
        u_max: input constraint bound
        x_max: dict {state_index: bound} for state constraints
        s_goal: terminal goal state (numpy array, shape (4,))

    Returns:
        problem: cp.Problem
        s_vars: list of n+1 state variables in error coordinates
        u_vars: list of n input variables
    """
    n_s = A.shape[0]
    n_u = B.shape[1]
    e0 = s0 - s_goal
    _validate_error_coords(s_goal, x_max)

    s_vars = [e0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
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

    # State constraints (valid because s_goal is zero at constrained indices)
    if x_max is not None:
        for idx, bound in x_max.items():
            constraints += [
                s_vars[k + 1][idx] <= bound for k in range(n)
            ] + [
                s_vars[k + 1][idx] >= -bound for k in range(n)
            ]

    # Terminal equality: error must reach zero
    constraints += [s_vars[n] == np.zeros(n_s)]

    problem = cp.Problem(objective, constraints)
    return problem, s_vars, u_vars


def create_invpend_rep_clqr(n, s0, A, B, Q, R, u_max, x_max, s_goal):
    """Rep method: dt as direct CVXPY parameters (Euler dynamics).

    Same structure as create_pann_param_clqr_2, with added state + terminal
    constraints and error-coordinate formulation.

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        A, B: continuous-time system matrices
        Q, R: cost matrices
        u_max: input constraint bound
        x_max: dict {state_index: bound} for state constraints
        s_goal: terminal goal state (numpy array)

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state variables in error coordinates
        u_vars: list of n input variables
        dts_param: single cp.Parameter of shape (n,)
    """
    n_s = A.shape[0]
    n_u = B.shape[1]
    e0 = s0 - s_goal
    _validate_error_coords(s_goal, x_max)

    s_vars = [e0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
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

    constraints += [s_vars[n] == np.zeros(n_s)]

    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[dts_param],
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, dts_param


def create_invpend_zoh_clqr(n, s0, n_s, n_u, u_max, x_max, s_goal):
    """ZOH3 method: exact ZOH dynamics + exact integrated quadratic cost.

    Same structure as create_exact_zoh_cost_clqr, with added state + terminal
    constraints and error-coordinate formulation.

    The Cholesky factor L of the cost matrix W is split into Lx (state columns)
    and Lu (input columns) so the expression Lx @ x_k + Lu @ u_k avoids
    cp.hstack on 1-D variables.

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        n_s: state dimension
        n_u: input dimension
        u_max: input constraint bound
        x_max: dict {state_index: bound} for state constraints
        s_goal: terminal goal state (numpy array)

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state variables in error coordinates
        u_vars: list of n input variables
        Aps: list of n Ad parameter matrices
        Bps: list of n Bd parameter matrices
        Lxs: list of n Lx parameter matrices
        Lus: list of n Lu parameter matrices
    """
    e0 = s0 - s_goal
    _validate_error_coords(s_goal, x_max)

    s_vars = [e0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
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

    constraints += [s_vars[n] == np.zeros(n_s)]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=Aps + Bps + Lxs + Lus,
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, Aps, Bps, Lxs, Lus
