#!/usr/bin/env python
"""Data generation for stiff system LTI differentiable time optimization.

Defines the example's `ProblemSpec` and delegates training to `training.py`.

Usage:
    python stiff_sys_train.py --mode test --experiment all
    python stiff_sys_train.py --mode full --experiment methods --method rep
    python stiff_sys_train.py --mode full --experiment losses --loss L_IV L_FI
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stiff_sys_prob import (
    create_stiff_sys_rep_clqr,
    create_stiff_sys_zoh_clqr,
    A, B, s0, T, n_default, Q, R, u_max, x_max, n_s, n_u,
)
from training import ProblemSpec, dispatch_main, train_custom_loss  # noqa: F401


def _rep_factory(n):
    _, layer, _, _, _ = create_stiff_sys_rep_clqr(
        n, s0, A, B, Q, R, u_max, x_max,
    )
    return layer, None


def _zoh_factory(n):
    _, layer, _, _, zoh_spec = create_stiff_sys_zoh_clqr(
        n, s0, n_s, n_u, u_max, x_max,
    )
    return layer, zoh_spec


SPEC = ProblemSpec(
    name="stiff_sys",
    A=A, B=B, Q=Q, R=R,
    s0=s0, s0_loss=s0,
    T=T, n_default=n_default,
    n_s=n_s, n_u=n_u,
    u_max=u_max, x_max=x_max,
    rep_factory=_rep_factory,
    zoh_factory=_zoh_factory,
    methods_available=["rep", "zoh"],
    default_n={"rep": 40, "zoh": 20},
    default_lr={"rep": 1e-2, "zoh": 1e-2},
    data_dirname="stiff_sys_dt",
)


if __name__ == "__main__":
    dispatch_main(SPEC)
