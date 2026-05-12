"""Regularizer losses driving non-uniform timestep distributions."""

import torch

from .discretization import zoh_discretize


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
