"""Shared training infrastructure for differentiable time optimization examples.

Each example (`cartpole`, `pann`, `stiff_sys`) defines a `ProblemSpec` and the
heavy lifting (training loops, argparse, pickle naming) is performed here.

Public entry points:
    train_softmax_method(spec, method, n, n_epochs, lr, data_dir)
    train_aux(spec, n, n_epochs, lr, data_dir)
    train_one_loss(spec, loss_name, n, n_epochs, lr, lambda0, use_balancing, data_dir, *, detach="none", disc=None)
    train_custom_loss(spec, loss_weights, n=None, n_epochs=200, lr=3e-2, **kwargs)
    dispatch_main(spec)  # argparse + dispatch
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from dito import (
    LOSS_REGISTRY,
    AdaptiveGradientBalancer,
    RAdaptDriver,
    build_loss_kwargs,
    euler_matrices,
    substep_loss,
    task_loss,
    theta_2_dt,
    uniform_resampling_loss,
    zoh_cost_matrices,
)
from utils import (
    RunMode,
    get_n_epochs,
    load_losses_config,
    load_pickle,
    pickle_name,
    resolve_loss_names,
    save_dts_distribution,
    save_pickle,
    save_run_config,
)


# ============================================================================ #
# Spec
# ============================================================================ #

# Mapping from public method name to internal-method-key recorded in history dicts.
_INTERNAL_METHOD_KEY = {
    "rep": "time_scaled",
    "rep_pann": "time scaled",          # pann uses a space (legacy)
    "hs_uniform": "uniform_resample",
    "hs_substeps": "substeps",
    "zoh": "exact_zoh_integrated",
}


@dataclass
class ProblemSpec:
    """Per-example training configuration.

    The factories are closures that capture system-level constants from the
    `<exp>_prob.py` module. Each factory takes only the runtime arguments
    needed by the shared training loops.
    """
    name: str
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    s0: np.ndarray
    s0_loss: np.ndarray              # state used at k=0 in the cost (s0 or e0)
    T: float
    n_default: int                   # default n for losses + reference for pickle suffix
    n_s: int
    n_u: int
    u_max: float
    x_max: Any                       # None | dict | float
    rep_factory: Callable[[int], Any]    # n -> (layer, None)
    zoh_factory: Callable[[int], Any]    # n -> (layer, ZOHParamSpec)
    methods_available: list[str]
    default_n: dict[str, int]
    default_lr: dict[str, float]
    data_dirname: str
    # Optional pann-only knobs
    aux_factory: Callable | None = None             # (n, s_bar, u_bar) -> 6-tuple
    pickle_n_default: int | None = None             # None disables _nXX suffix logic
    use_lr_scheduler: bool = False
    grad_clip_norm: float | None = None
    loss_disc_choices: tuple[str, ...] = ("zoh",)
    loss_disc_default: str = "zoh"
    supports_detach: bool = True
    use_pann_rep_internal_key: bool = False         # legacy "time scaled" with space
    # Internal key mapping (auto-derived from spec)
    internal_method_key: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.internal_method_key:
            self.internal_method_key = dict(_INTERNAL_METHOD_KEY)
            if self.use_pann_rep_internal_key:
                self.internal_method_key["rep"] = "time scaled"


# ============================================================================ #
# Helpers
# ============================================================================ #

def make_torch_constants(spec: ProblemSpec, dtype: torch.dtype):
    """Convert system constants to torch tensors."""
    A_t = torch.tensor(spec.A, dtype=dtype)
    B_t = torch.tensor(spec.B, dtype=dtype)
    Q_t = torch.tensor(spec.Q, dtype=dtype)
    R_t = torch.tensor(spec.R, dtype=dtype)
    s0_loss_t = torch.tensor(spec.s0_loss, dtype=dtype)
    return A_t, B_t, Q_t, R_t, s0_loss_t


def make_tau_at(tau_schedule, n_epochs):
    """Return a callable epoch -> tau, given an optional schedule tuple.

    schedule = None         -> constant tau=1
    schedule = (kind, t0, t1) with kind in {"linear","exp"} -> annealing.
    """
    if tau_schedule is None:
        return lambda epoch: 1.0

    kind, tau_start, tau_end = tau_schedule
    if kind not in ("linear", "exp"):
        raise ValueError(
            f"tau_schedule kind must be 'linear' or 'exp', got '{kind}'")
    if tau_start <= 0 or tau_end <= 0:
        raise ValueError("tau_start and tau_end must be positive")

    if kind == "linear":
        def tau_at(epoch):
            if n_epochs <= 1:
                return float(tau_end)
            t = epoch / (n_epochs - 1)
            return float(tau_start + (tau_end - tau_start) * t)
    else:
        log_start = np.log(tau_start)
        log_end = np.log(tau_end)

        def tau_at(epoch):
            if n_epochs <= 1:
                return float(tau_end)
            t = epoch / (n_epochs - 1)
            return float(np.exp(log_start + (log_end - log_start) * t))

    return tau_at


def _create_layer(spec: ProblemSpec, method: str, n: int):
    """Return (layer, zoh_spec_or_None) for a softmax-based method."""
    if method in ("rep", "hs_uniform", "hs_substeps"):
        return spec.rep_factory(n)
    if method == "zoh":
        return spec.zoh_factory(n)
    raise ValueError(f"Unknown softmax method: {method}")


def _compute_qp_params_and_solve(spec, method, layer, zoh_spec, dts_torch, n,
                                 A_t, B_t, Q_t, R_t):
    """Solve the QP for one method. Returns (sol, W_list_or_None)."""
    if method in ("rep", "hs_uniform", "hs_substeps"):
        return layer(dts_torch), None
    if method == "zoh":
        packed_steps, W_list = [], []
        for k in range(n):
            Ad_k, Bd_k, W_k = zoh_cost_matrices(dts_torch[k], A_t, B_t, Q_t, R_t)
            W_list.append(W_k)
            packed_steps.append(zoh_spec.pack_step(Ad_k, Bd_k, W_k))
        return layer(*zoh_spec.flatten_for_layer(packed_steps)), W_list
    raise ValueError(f"Unknown method: {method}")


def _compute_loss(spec, method, sol, dts_torch, n, W_list, s0_loss_t,
                  A_t, B_t, Q_t, R_t):
    """Compute the training loss for a softmax-based method."""
    dtype = dts_torch.dtype

    if method == "rep":
        states_sol = [sol[i].to(dtype) for i in range(n)]
        inputs_sol = [sol[n + i].to(dtype) for i in range(n)]
        return task_loss(states_sol, inputs_sol, dts_torch, spec.Q, spec.R,
                         method="time_scaled")

    if method == "hs_uniform":
        inputs_qp = [sol[n + i].to(dtype) for i in range(n)]
        return uniform_resampling_loss(
            inputs_qp, dts_torch, s0_loss_t, A_t, B_t, Q_t, R_t,
            T=spec.T, n_res=1000, use_exact=False,
        )

    if method == "hs_substeps":
        inputs_qp = [sol[n + i].to(dtype) for i in range(n)]
        return substep_loss(
            inputs_qp, dts_torch, s0_loss_t, A_t, B_t, Q_t, R_t,
            n_sub=10, use_exact=False,
        )

    if method == "zoh":
        loss = torch.tensor(0.0, dtype=dtype)
        for k in range(n):
            s_k = s0_loss_t if k == 0 else sol[k - 1].to(dtype)
            u_k = sol[n + k].to(dtype)
            z_k = torch.cat([s_k, u_k])
            loss = loss + z_k @ W_list[k] @ z_k
        return loss

    raise ValueError(f"Unknown method: {method}")


def _loss_kwargs(spec, name, states, inputs, dts_torch, W_list, Ad_list,
                 Bd_list, A_t, B_t, Q_t, R_t):
    """Build kwargs for build_loss_kwargs, threading x_max through when set."""
    return build_loss_kwargs(
        name, states, inputs, dts_torch, W_list, Ad_list, Bd_list,
        A_t, B_t, Q_t, R_t, T=spec.T, u_max=spec.u_max, x_max=spec.x_max,
    )


def _pickle_names(spec, method, n):
    """Compute (sol, history, dts_dist) pickle base names for a method."""
    if method in ("hs_uniform", "hs_substeps"):
        internal = _INTERNAL_METHOD_KEY[method]
        return ("sol_hs", "history_hs", f"dts_dist_hs_{internal}")
    if method == "rep" and "hs_uniform" in spec.methods_available:
        # Pann historical naming: rep dts_dist gets internal-key suffix.
        return ("sol_rep", "history_rep", "dts_dist_rep_time_scaled")
    if method == "zoh" and spec.pickle_n_default is not None:
        return (
            pickle_name("sol_zoh", n, spec.pickle_n_default),
            pickle_name("history_zoh", n, spec.pickle_n_default),
            pickle_name("dts_dist_zoh", n, spec.pickle_n_default),
        )
    return f"sol_{method}", f"history_{method}", f"dts_dist_{method}"


# ============================================================================ #
# Method training: aux (pann only)
# ============================================================================ #

def train_aux(spec: ProblemSpec, n, n_epochs, lr, data_dir):
    """Auxiliary-variable method (delta_k as QP variables). Pann-only."""
    if spec.aux_factory is None:
        raise RuntimeError(f"Spec '{spec.name}' has no aux_factory")

    Q_torch = torch.tensor(spec.Q, dtype=torch.float64)
    R_torch = torch.tensor(spec.R, dtype=torch.float64)

    loss_methods = ["time scaled", "time bar scaled"]
    history_aux = []
    sol_aux = {}

    for method in loss_methods:
        print(f"  Aux sub-method: {method}")

        dt_init = spec.T / n
        dts_torch = [torch.nn.Parameter(torch.ones(1) * dt_init) for _ in range(n)]
        optim = torch.optim.Adam(dts_torch, lr=lr)

        with torch.no_grad():
            for d in dts_torch:
                d.copy_(torch.ones(1) * dt_init)

        s_bar = [spec.s0 + (np.zeros(spec.n_s) - spec.s0) * i / n for i in range(n)]
        u_bar = [np.zeros(spec.n_u) for _ in range(n)]

        with tqdm(total=n_epochs, desc=f"aux ({method})") as pbar:
            for epoch in range(n_epochs):
                pbar.update(1)
                optim.zero_grad()

                _, layer_aux, _, _, _, _ = spec.aux_factory(n, s_bar, u_bar)
                sol_aux[method] = layer_aux(*dts_torch)

                s_bar = [sol_aux[method][i].detach().numpy() for i in range(n)]
                u_bar = [sol_aux[method][n + i].detach().numpy() for i in range(n)]

                states_sol = [sol_aux[method][i] for i in range(n)]
                inputs_sol = [sol_aux[method][n + i] for i in range(n)]
                deltas_sol = [sol_aux[method][2 * n + i] for i in range(n)]

                if method == "time scaled":
                    loss = sum(
                        deltas_sol[i] * states_sol[i].t() @ Q_torch @ states_sol[i]
                        for i in range(n)
                    ) + sum(
                        deltas_sol[i] * inputs_sol[i].t() @ R_torch @ inputs_sol[i]
                        for i in range(n)
                    )
                elif method == "time bar scaled":
                    deltas_bar = np.concatenate(
                        [d.detach().numpy() for d in dts_torch])
                    loss = sum(
                        deltas_bar[i] * states_sol[i].t() @ Q_torch @ states_sol[i]
                        for i in range(n)
                    ) + sum(
                        deltas_bar[i] * inputs_sol[i].t() @ R_torch @ inputs_sol[i]
                        for i in range(n)
                    )
                else:
                    loss = task_loss(states_sol, inputs_sol, dts_torch,
                                     spec.Q, spec.R, method="unscaled")

                loss.backward()
                optim.step()

                with torch.no_grad():
                    for d in dts_torch:
                        d.clamp_(min=1e-6, max=0.07)
                        d *= spec.T / sum(dts_torch)

                history_aux.append({
                    'method': method,
                    'loss': loss.item(),
                    'dts': [d.detach().numpy() for d in dts_torch],
                })

    save_pickle(data_dir, "sol_aux", sol_aux)
    save_pickle(data_dir, "history_aux", history_aux)
    for method in loss_methods:
        hist_m = [h for h in history_aux if h['method'] == method]
        _key = method.replace(' ', '_')
        save_dts_distribution(data_dir, f"dts_dist_aux_{_key}", hist_m)

    return sol_aux, history_aux


# ============================================================================ #
# Method training: softmax (rep, hs_uniform, hs_substeps, zoh)
# ============================================================================ #

def train_softmax_method(spec: ProblemSpec, method, n, n_epochs, lr, data_dir):
    """Train a softmax-parametrized method. Returns (sol, history)."""
    internal_key = spec.internal_method_key[method]

    dtype = torch.float32
    A_t, B_t, Q_t, R_t, s0_loss_t = make_torch_constants(spec, dtype)

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    layer, zoh_spec = _create_layer(spec, method, n)

    history = []
    sol_dict = {}

    with tqdm(total=n_epochs, desc=method) as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            dts_torch = theta_2_dt(theta, spec.T, n)
            sol_raw, W_list = _compute_qp_params_and_solve(
                spec, method, layer, zoh_spec, dts_torch, n,
                A_t, B_t, Q_t, R_t,
            )
            sol_dict[internal_key] = sol_raw

            loss = _compute_loss(
                spec, method, sol_raw, dts_torch, n, W_list, s0_loss_t,
                A_t, B_t, Q_t, R_t,
            )
            loss.backward()
            optim.step()

            history.append({
                'method': internal_key,
                'epoch': epoch,
                'loss': float(loss.item()),
                'dts': dts_torch.detach().cpu().numpy(),
            })

    sol_name, hist_name, dist_name = _pickle_names(spec, method, n)

    sol_to_save = sol_dict
    history_to_save = history
    if method in ("hs_uniform", "hs_substeps"):
        # Pann's hs_* methods share sol_hs/history_hs pickles — merge.
        try:
            existing_sol = load_pickle(data_dir, sol_name)
            existing_hist = load_pickle(data_dir, hist_name)
            existing_sol[internal_key] = sol_dict[internal_key]
            existing_hist.extend(history)
            sol_to_save = existing_sol
            history_to_save = existing_hist
        except (FileNotFoundError, OSError):
            pass

    save_pickle(data_dir, sol_name, sol_to_save)
    save_pickle(data_dir, hist_name, history_to_save)
    save_dts_distribution(data_dir, dist_name, history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol_to_save, history


# ============================================================================ #
# Loss training: ZOH/FOE base + single regularizer
# ============================================================================ #

def train_one_loss(spec: ProblemSpec, loss_name, n, n_epochs, lr, lambda0,
                   use_balancing, data_dir, *, detach="none", disc=None):
    """Train one alternative loss as a regularizer on the ZOH-cost OCP.

    Args:
        detach: gradient detach mode for the QP solution.
            "none"  -- full gradient through cvxpylayers
            "reg"   -- detach states/inputs for L_reg only
            "all"   -- detach for both L_ocp and L_reg
        disc: discretization for cost matrices ("zoh" or "foe"). Defaults to
            spec.loss_disc_default.
    """
    if disc is None:
        disc = spec.loss_disc_default
    if disc not in spec.loss_disc_choices:
        raise ValueError(
            f"disc must be one of {spec.loss_disc_choices}, got '{disc}'")

    dtype = torch.float64
    A_t, B_t, Q_t, R_t, s0_loss_t = make_torch_constants(spec, dtype)

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)
    scheduler = None
    if spec.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=n_epochs, eta_min=lr * 0.01)

    layer, zoh_spec = spec.zoh_factory(n)

    balancer = (AdaptiveGradientBalancer(lambda_0=lambda0)
                if use_balancing else None)
    loss_fn = LOSS_REGISTRY[loss_name]

    compute_matrices = zoh_cost_matrices if disc == "zoh" else euler_matrices

    history = []
    sol = None

    with tqdm(total=n_epochs, desc=loss_name) as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            dts_torch = theta_2_dt(theta, spec.T, n)

            packed_steps, Ad_list, Bd_list, W_list = [], [], [], []
            for k in range(n):
                Ad_k, Bd_k, W_k = compute_matrices(
                    dts_torch[k], A_t, B_t, Q_t, R_t)
                Ad_list.append(Ad_k)
                Bd_list.append(Bd_k)
                W_list.append(W_k)
                packed_steps.append(zoh_spec.pack_step(Ad_k, Bd_k, W_k))

            sol = layer(*zoh_spec.flatten_for_layer(packed_steps))

            states = [s0_loss_t] + [sol[k] for k in range(n)]
            inputs = [sol[n + k] for k in range(n)]

            if detach in ("reg", "all"):
                states_d = [s.detach() for s in states]
                inputs_d = [u.detach() for u in inputs]
            states_ocp = states_d if detach == "all" else states
            inputs_ocp = inputs_d if detach == "all" else inputs
            states_reg = states_d if detach in ("reg", "all") else states
            inputs_reg = inputs_d if detach in ("reg", "all") else inputs

            loss_ocp = torch.tensor(0.0, dtype=dtype)
            for k in range(n):
                z_k = torch.cat([states_ocp[k], inputs_ocp[k]])
                loss_ocp = loss_ocp + z_k @ W_list[k] @ z_k

            kwargs = _loss_kwargs(
                spec, loss_name, states_reg, inputs_reg, dts_torch, W_list,
                Ad_list, Bd_list, A_t, B_t, Q_t, R_t,
            )
            loss_reg = loss_fn(**kwargs)

            if balancer is not None:
                lambda_hat = balancer.step(theta, loss_ocp, loss_reg)
            else:
                lambda_hat = lambda0

            loss = loss_ocp + lambda_hat * loss_reg
            loss.backward()
            if spec.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [theta], max_norm=spec.grad_clip_norm)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            entry = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_ocp": float(loss_ocp.item()),
                "loss_reg": float(loss_reg.item()),
                "lambda_hat": float(lambda_hat),
                "dts": dts_torch.detach().cpu().numpy(),
            }
            if spec.supports_detach:
                entry["detach"] = detach
            history.append(entry)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ocp=f"{loss_ocp.item():.4f}",
                reg=f"{loss_reg.item():.4f}",
                lam=f"{lambda_hat:.4f}",
            )

    save_pickle(data_dir, f"sol_{loss_name}", sol)
    save_pickle(data_dir, f"history_{loss_name}", history)
    save_dts_distribution(data_dir, f"dts_dist_{loss_name}", history)

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history


# ============================================================================ #
# Custom composite loss training (notebook entry point)
# ============================================================================ #

def train_custom_loss(spec: ProblemSpec, loss_weights, n=None, n_epochs=200,
                      lr=3e-2, *, detach="none", discretization="zoh",
                      ocp_weight=1.0, tau_schedule=None,
                      radapt_enable=False,
                      radapt_every=10,
                      radapt_freq_schedule=None,
                      radapt_importance="combined",
                      radapt_beta=1.0,
                      radapt_temp_schedule=("exp", 1.0, 0.01),
                      radapt_warmup=20,
                      radapt_tail_skip=20,
                      radapt_accept_cooldown=0,
                      radapt_seed=None):
    """Training loop with a fixed-weight composite loss.

    Total loss is  ocp_weight * L_ocp + sum_i (w_i * L_i).
    """
    if discretization not in ("zoh", "euler"):
        raise ValueError(
            f"discretization must be 'zoh' or 'euler', got '{discretization}'")

    tau_at = make_tau_at(tau_schedule, n_epochs)

    n = n or spec.n_default
    dtype = torch.float64
    A_t, B_t, Q_t, R_t, s0_loss_t = make_torch_constants(spec, dtype)

    for name in loss_weights:
        if name not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    theta = torch.nn.Parameter(torch.ones(n, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=lr)

    if discretization == "zoh":
        layer, zoh_spec = spec.zoh_factory(n)
    else:
        layer, _ = spec.rep_factory(n)
        zoh_spec = None

    disc_fn = zoh_cost_matrices if discretization == "zoh" else euler_matrices
    loss_fns = {name: LOSS_REGISTRY[name] for name in loss_weights}

    radapt = RAdaptDriver(
        n_epochs=n_epochs, Q=spec.Q, R=spec.R,
        enable=radapt_enable, every=radapt_every,
        freq_schedule=radapt_freq_schedule,
        importance=radapt_importance, beta=radapt_beta,
        temp_schedule=radapt_temp_schedule,
        warmup=radapt_warmup, tail_skip=radapt_tail_skip,
        accept_cooldown=radapt_accept_cooldown,
        seed=radapt_seed,
    )

    def _evaluate(epoch_):
        tau_l = tau_at(epoch_)
        theta_eff_l = theta if tau_l == 1.0 else theta / tau_l
        dts_l = theta_2_dt(theta_eff_l, spec.T, n)

        Ad_l, Bd_l, W_l = [], [], []
        packed_l = []
        for k in range(n):
            Ad_k, Bd_k, W_k = disc_fn(dts_l[k], A_t, B_t, Q_t, R_t)
            Ad_l.append(Ad_k)
            Bd_l.append(Bd_k)
            W_l.append(W_k)
            if discretization == "zoh":
                packed_l.append(zoh_spec.pack_step(Ad_k, Bd_k, W_k))

        if discretization == "zoh":
            sol_l = layer(*zoh_spec.flatten_for_layer(packed_l))
        else:
            sol_l = layer(dts_l)

        states_l = [s0_loss_t] + [sol_l[k] for k in range(n)]
        inputs_l = [sol_l[n + k] for k in range(n)]

        if detach in ("reg", "all"):
            states_d = [s.detach() for s in states_l]
            inputs_d = [u.detach() for u in inputs_l]
        states_ocp = states_d if detach == "all" else states_l
        inputs_ocp = inputs_d if detach == "all" else inputs_l
        states_reg = states_d if detach in ("reg", "all") else states_l
        inputs_reg = inputs_d if detach in ("reg", "all") else inputs_l

        loss_ocp_l = torch.tensor(0.0, dtype=dtype)
        for k in range(n):
            z_k = torch.cat([states_ocp[k], inputs_ocp[k]])
            loss_ocp_l = loss_ocp_l + z_k @ W_l[k] @ z_k

        reg_losses_l = {}
        loss_reg_total_l = torch.tensor(0.0, dtype=dtype)
        for name, w in loss_weights.items():
            kwargs = _loss_kwargs(
                spec, name, states_reg, inputs_reg, dts_l, W_l,
                Ad_l, Bd_l, A_t, B_t, Q_t, R_t,
            )
            l_reg = loss_fns[name](**kwargs)
            reg_losses_l[name] = l_reg
            loss_reg_total_l = loss_reg_total_l + w * l_reg

        loss_l = ocp_weight * loss_ocp_l + loss_reg_total_l

        return {
            "loss": loss_l, "loss_ocp": loss_ocp_l,
            "loss_reg_total": loss_reg_total_l,
            "reg_losses": reg_losses_l, "sol": sol_l,
            "states": states_l, "inputs": inputs_l, "dts": dts_l, "tau": tau_l,
        }

    history = []
    sol = None

    label = " + ".join(f"{w}*{name}" for name, w in loss_weights.items())
    disc_tag = discretization.upper()
    with tqdm(total=n_epochs, desc=f"custom/{disc_tag} ({label})") as pbar:
        for epoch in range(n_epochs):
            pbar.update(1)
            optim.zero_grad(set_to_none=True)

            out = _evaluate(epoch)
            loss = out["loss"]
            loss.backward()
            optim.step()
            sol = out["sol"]

            entry = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_ocp": float(out["loss_ocp"].item()),
                "loss_reg_total": float(out["loss_reg_total"].item()),
                "dts": out["dts"].detach().cpu().numpy(),
                "tau": out["tau"],
                "detach": detach,
            }
            entry.update({f"loss_{name}": float(v.item())
                          for name, v in out["reg_losses"].items()})

            entry.update(radapt.maybe_step(epoch, theta, _evaluate, optim))
            history.append(entry)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ocp=f"{out['loss_ocp'].item():.4f}",
                reg=f"{out['loss_reg_total'].item():.4f}",
                tau=f"{out['tau']:.3f}",
            )

    print(f"  Final loss: {history[-1]['loss']:.6f}")
    return sol, history, n


# ============================================================================ #
# Argparse + dispatch
# ============================================================================ #

def build_train_argparser(spec: ProblemSpec):
    """Build the CLI argparser for an example. Choices come from the spec."""
    parser = argparse.ArgumentParser(
        description=f"Data generation for {spec.name} differentiable time optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["test", "full"])
    parser.add_argument("--experiment", required=True,
                        choices=["methods", "losses", "all"])
    parser.add_argument(
        "--method", nargs="+", default=["all"],
        help=f"Method(s) to train, or 'all'. Available: {spec.methods_available}",
    )
    parser.add_argument(
        "--loss", nargs="+", default=["default"],
        help="Loss function(s). Use 'default' (config's enabled list) or 'all'. "
             f"Available: {list(LOSS_REGISTRY.keys())}",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda0", type=float, default=0.3)
    parser.add_argument("--no-balancing", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None, metavar="PATH")
    if spec.supports_detach:
        parser.add_argument(
            "--detach", choices=["none", "reg", "all"], default="none",
            help="Detach QP solution from gradient graph",
        )
    if len(spec.loss_disc_choices) > 1:
        parser.add_argument(
            "--disc", choices=spec.loss_disc_choices,
            default=spec.loss_disc_default,
            help="Discretization for the loss-training cost matrices",
        )
    return parser


def dispatch_main(spec: ProblemSpec):
    """Run the full CLI flow for an example."""
    parser = build_train_argparser(spec)
    args = parser.parse_args()

    run_mode = RunMode.TEST if args.mode == "test" else RunMode.FULL
    losses_cfg = load_losses_config(args.config)

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", spec.data_dirname)
    os.makedirs(data_dir, exist_ok=True)

    save_run_config(data_dir, args)

    print(f"Output directory: {data_dir}")
    print(f"Mode: {args.mode}")
    print()

    detach = getattr(args, "detach", "none")
    disc = getattr(args, "disc", spec.loss_disc_default)

    # Methods
    if args.experiment in ("methods", "all"):
        methods = spec.methods_available if "all" in args.method else args.method
        for m in methods:
            if m not in spec.methods_available:
                parser.error(
                    f"Unknown method: {m}. Available: {spec.methods_available}")

        print(f"Methods: {methods}")
        for method_name in methods:
            n = args.n or spec.default_n[method_name]
            lr = args.lr or spec.default_lr[method_name]
            epoch_key = method_name
            if method_name in ("hs_uniform", "hs_substeps"):
                epoch_key = "hs"
            n_epochs = args.epochs or get_n_epochs(run_mode, epoch_key)

            print(f"\n{'=' * 40}")
            print(f"Training {method_name} (n={n}, epochs={n_epochs}, lr={lr})")
            print(f"{'=' * 40}")

            if method_name == "aux":
                train_aux(spec, n, n_epochs, lr, data_dir)
            else:
                train_softmax_method(spec, method_name, n, n_epochs, lr, data_dir)

    # Losses
    if args.experiment in ("losses", "all"):
        loss_names = resolve_loss_names(args.loss, losses_cfg)
        if spec.x_max is None:
            loss_names = [l for l in loss_names if l != "L_SC"]
        for name in loss_names:
            if name not in LOSS_REGISTRY:
                parser.error(
                    f"Unknown loss: {name}. "
                    f"Available: {list(LOSS_REGISTRY.keys())}")

        n_loss_default = spec.n_default
        lr_loss = args.lr or 3e-2
        per_loss_cfg = losses_cfg.get("per_loss", {}) if losses_cfg else {}
        balance_str = "adaptive" if not args.no_balancing else "fixed"
        info = (f"\nLosses: {loss_names}\nBalancing: {balance_str}, "
                f"lambda0={args.lambda0}")
        if spec.supports_detach:
            info += f", detach={detach}"
        if len(spec.loss_disc_choices) > 1:
            info += f", disc={disc}"
        print(info)

        for loss_name in loss_names:
            n = args.n or n_loss_default
            n_epochs = args.epochs or get_n_epochs(run_mode, loss_name)
            lambda0 = per_loss_cfg.get(loss_name, {}).get(
                "lambda0", args.lambda0)

            print(f"\n{'=' * 40}")
            print(f"Training {loss_name} ({n_epochs} epochs, lambda0={lambda0})")
            print(f"{'=' * 40}")

            train_one_loss(
                spec, loss_name, n, n_epochs, lr_loss, lambda0,
                not args.no_balancing, data_dir,
                detach=detach, disc=disc,
            )

    print(f"\nResults saved to: {data_dir}")
