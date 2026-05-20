"""Phase 3 numerics probe: same-seed cost / theta.grad norm at (n_s=5, n_c=20).

Compare against Phase 2 reference values from CLAUDE.md:
    cost = 0.6283273994
    theta.grad norm = 0.4516235925   (post-Phase-2 value)
"""
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(TOP_DIR, "time_optimization", "stiff_sys"))

from stiff_sys_times import build_stiff_system  # noqa: E402

from dito import ZOHParamSpec, theta_2_dt, zoh_cost_matrices  # noqa: E402
import cvxpy as cp  # noqa: E402
from cvxpylayers.torch import CvxpyLayer  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

DTYPE = torch.float64
N_S = 5
N_C = 20
T = 10.0
U_MAX = 10.0


def main():
    A_np, B_np, Q_np, R_np, s0_np, _, n_u = build_stiff_system(N_S)
    spec = ZOHParamSpec(A_np, B_np, Q_np, R_np)

    s_vars = [s0_np] + [cp.Variable(N_S, name=f"s_{i}") for i in range(N_C)]
    u_vars = [cp.Variable(n_u, name=f"u_{i}") for i in range(N_C)]
    step_params = spec.make_step_params(N_C)

    objective = spec.total_cost_expr(s_vars[:N_C], u_vars, step_params)
    S_next = cp.vstack(s_vars[1:])
    U_mat = cp.vstack(u_vars)
    DYN = cp.vstack([
        spec.dynamics_expr(s_vars[k], u_vars[k], step_params[k])
        for k in range(N_C)
    ])
    constraints = [S_next == DYN, cp.abs(U_mat) <= U_MAX]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    layer = CvxpyLayer(
        problem,
        parameters=spec.layer_parameters(step_params),
        variables=s_vars[1:] + u_vars,
        canon_backend=cp.COO_CANON_BACKEND,
    )

    A_t = torch.tensor(A_np, dtype=DTYPE)
    B_t = torch.tensor(B_np, dtype=DTYPE)
    Q_t = torch.tensor(Q_np, dtype=DTYPE)
    R_t = torch.tensor(R_np, dtype=DTYPE)

    theta = torch.nn.Parameter(torch.ones(N_C, 1, dtype=DTYPE))
    dts = theta_2_dt(theta, T, N_C)
    solver_args = {"mode": "lsqr", "solve_method": "CLARABEL"}

    packed, W_list = [], []
    for k in range(N_C):
        Ad_k, Bd_k, W_k = zoh_cost_matrices(dts[k], A_t, B_t, Q_t, R_t)
        W_list.append(W_k)
        packed.append(spec.pack_step(Ad_k, Bd_k, W_k))

    sol = layer(*spec.flatten_for_layer(packed), solver_args=solver_args)

    s0_t = torch.tensor(s0_np, dtype=DTYPE)
    cost = torch.tensor(0.0, dtype=DTYPE)
    for k in range(N_C):
        s_k = s0_t if k == 0 else sol[k - 1].to(DTYPE)
        u_k = sol[N_C + k].to(DTYPE)
        z_k = torch.cat([s_k, u_k])
        cost = cost + z_k @ W_list[k] @ z_k

    cost.backward()

    grad_norm = theta.grad.norm().item()
    print(f"cost           = {cost.item():.10f}")
    print(f"theta.grad norm = {grad_norm:.10f}")
    print(f"sol[0][:5]      = {sol[0].detach()[:5].numpy()}")


if __name__ == "__main__":
    main()
