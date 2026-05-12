#!/usr/bin/env python
"""Data generation for Pannocchia CLQR differentiable time optimization.

Trains methods (aux, rep, hs_uniform, hs_substeps, zoh) and alternative loss
functions for learning non-uniform timesteps in constrained LQR.

Usage:
    python pann_train.py --mode test --experiment all
    python pann_train.py --mode full --experiment methods --method rep zoh
    python pann_train.py --mode full --experiment losses --loss L_IV L_FI
    python pann_train.py --mode full --experiment methods --method zoh --n 80 --epochs 500
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pann_prob import (
    create_pann_param_clqr,
    create_pann_param_clqr_2,
    create_exact_zoh_cost_clqr,
    A, B, s0, T, Q, R, u_max, n_s, n_u,
)
from training import ProblemSpec, dispatch_main, train_custom_loss  # noqa: F401


def _rep_factory(n):
    _, layer, _, _, _ = create_pann_param_clqr_2(n, s0, A, B, Q, R, u_max)
    return layer, None


def _zoh_factory(n):
    _, layer, _, _, zoh_spec = create_exact_zoh_cost_clqr(
        n, s0, n_s, n_u, u_max,
    )
    return layer, zoh_spec


def _aux_factory(n, s_bar, u_bar):
    return create_pann_param_clqr(n, s0, A, B, Q, R, s_bar, u_bar, u_max, T)


SPEC = ProblemSpec(
    name="pann",
    A=A, B=B, Q=Q, R=R,
    s0=s0, s0_loss=s0,
    T=T, n_default=160,
    n_s=n_s, n_u=n_u,
    u_max=u_max, x_max=None,
    rep_factory=_rep_factory,
    zoh_factory=_zoh_factory,
    aux_factory=_aux_factory,
    methods_available=["aux", "rep", "hs_uniform", "hs_substeps", "zoh"],
    default_n={"aux": 160, "rep": 160, "hs_uniform": 160, "hs_substeps": 160,
               "zoh": 80},
    default_lr={"aux": 5e-4, "rep": 1e-2, "hs_uniform": 1e-2,
                "hs_substeps": 1e-2, "zoh": 1e-2},
    data_dirname="pann_clqr_dt",
    pickle_n_default=160,
    use_lr_scheduler=True,
    grad_clip_norm=1.0,
    loss_disc_choices=("foe", "zoh"),
    loss_disc_default="foe",
    supports_detach=False,
    use_pann_rep_internal_key=True,
)


if __name__ == "__main__":
    dispatch_main(SPEC)
