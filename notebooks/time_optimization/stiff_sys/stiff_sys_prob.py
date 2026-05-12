"""Stiff system LTI QP builders for differentiable time optimization.

The system has three states with well-separated time constants (0.1s, 10s, 100s),
making non-uniform timestep placement especially beneficial.

Each function creates its own local CVXPY variables and parameters (no globals)
and returns everything the caller needs, including a CvxpyLayer when applicable.

No error coordinates are needed since the goal state is the origin.
"""

import os
import sys

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dito import ZOHParamSpec  # noqa: E402

# ============================================================================ #
# System Constants (Stiff System LTI)
# ============================================================================ #

A = np.array([
    [-10,     0.0,    0.0],
    [0.0,    -0.1,    0.0],
    [0.0,     0.0,  -0.01],
])

B = np.array([
    [1.0],
    [1.0],
    [1.0],
])

s0 = np.array([-1.0, -1.0, -1.0])
T = 10.0
n_default = 40
Q = 1.0 * np.eye(3)
R = 0.01 * np.eye(1)
u_max = 10.0
x_max = None
n_s = 3
n_u = 1


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


def create_stiff_sys_zoh_clqr(n, s0, n_s, n_u, u_max, x_max,
                              A_sys=None, B_sys=None, Q_sys=None, R_sys=None):
    """ZOH3 method: exact ZOH dynamics + exact integrated quadratic cost.

    Uses ZOHParamSpec to declare structure-aware parameter shapes. The stiff
    system is jointly diagonal in A and Q, so detection lands in the
    fully-diagonal fast path of the block-diagonal tier.

    Args:
        n: number of timesteps
        s0: initial state (numpy array)
        n_s: state dimension
        n_u: input dimension
        u_max: input constraint bound (scalar)
        x_max: dict {state_index: bound} for state constraints, or None
        A_sys, B_sys, Q_sys, R_sys: continuous-time matrices used for structure
            detection. Default to the module-level constants. Override when
            building a problem for a system that differs from the module
            defaults (e.g., the n_states sweep in stiff_sys_times.py).

    Returns:
        problem: cp.Problem
        layer: CvxpyLayer
        s_vars: list of n+1 state variables
        u_vars: list of n input variables
        spec: ZOHParamSpec used to declare the per-step parameters
    """
    A_sys = A if A_sys is None else A_sys
    B_sys = B if B_sys is None else B_sys
    Q_sys = Q if Q_sys is None else Q_sys
    R_sys = R if R_sys is None else R_sys
    spec = ZOHParamSpec(A_sys, B_sys, Q_sys, R_sys)

    s_vars = [s0] + [cp.Variable(n_s, name=f"s_{i}") for i in range(n)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(n)]
    step_params = [spec.make_step_params(k) for k in range(n)]

    objective = cp.sum([
        spec.cost_expr(s_vars[k], u_vars[k], step_params[k])
        for k in range(n)
    ])

    constraints = [
        s_vars[k + 1] == spec.dynamics_expr(s_vars[k], u_vars[k], step_params[k])
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
        parameters=spec.layer_parameters(step_params),
        variables=s_vars[1:] + u_vars,
    )

    return problem, layer, s_vars, u_vars, spec
