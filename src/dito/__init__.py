"""DITO — Differentiable Time Optimization.

Library of differentiable primitives for learning non-uniform discretization
timesteps in optimal control via cvxpylayers.
"""

from .cvxpy_zoh import ZOHParamSpec
from .discretization import (
    Ad_Bd_from_dt,
    LQs_LRs_from_dt,
    euler_matrices,
    zoh_cost_matrices,
    zoh_discretize,
)
from .eval import (
    evaluate_continuous_cost,
    substep_loss,
    task_loss,
    uniform_resampling_loss,
)
from .grad_balance import AdaptiveGradientBalancer
from .losses import (
    LOSS_REGISTRY,
    build_loss_kwargs,
    loss_cpc,
    loss_css,
    loss_defect,
    loss_dyn,
    loss_eq,
    loss_equi,
    loss_fi,
    loss_iv,
    loss_iv_rate,
    loss_iv_sym,
    loss_pwlh,
    loss_sc,
    loss_ssd,
)
from .parametrization import theta_2_dt
from .plotting import (
    _draw_zoh_segments,
    _resolve_cmap_norm,
    add_zoom_inset,
    plot_colored,
    plot_timegrid,
)
from .r_adapt import (
    RAdaptDriver,
    apply_merge_split,
    compute_importance,
    select_merge_split,
)

__all__ = [
    "Ad_Bd_from_dt",
    "AdaptiveGradientBalancer",
    "LOSS_REGISTRY",
    "LQs_LRs_from_dt",
    "RAdaptDriver",
    "ZOHParamSpec",
    "_draw_zoh_segments",
    "_resolve_cmap_norm",
    "add_zoom_inset",
    "apply_merge_split",
    "build_loss_kwargs",
    "compute_importance",
    "euler_matrices",
    "evaluate_continuous_cost",
    "loss_cpc",
    "loss_css",
    "loss_defect",
    "loss_dyn",
    "loss_eq",
    "loss_equi",
    "loss_fi",
    "loss_iv",
    "loss_iv_rate",
    "loss_iv_sym",
    "loss_pwlh",
    "loss_sc",
    "loss_ssd",
    "plot_colored",
    "plot_timegrid",
    "select_merge_split",
    "substep_loss",
    "task_loss",
    "theta_2_dt",
    "uniform_resampling_loss",
    "zoh_cost_matrices",
    "zoh_discretize",
]
