"""Stiff-system speed benchmark: n_states in {5, 10, 15, 20, 25, 30} x n_control in {20, 40, 60, 80, 100}.

Uses the exact-ZOH parametrization (the same path the training scripts in
``notebooks/time_optimization/stiff_sys`` use). System matrices are built with
the same fixed |lambda_max|/|lambda_min| stiffness ratio as
``stiff_sys_times.py`` so cells of different ``n_states`` stay comparable.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import make_run_dir, time_zoh_cell, warmup, write_metadata
from stiff_sys_prob import create_stiff_sys_zoh_clqr  # noqa: E402

LAMBDA_MAX = 10.0
STIFFNESS_RATIO = 1000.0
T_HORIZON = 10.0
U_MAX = 10.0
X_MAX = None

N_STATES_LIST = [5, 10, 15, 20, 25, 30]
N_CONTROL_LIST = [20, 40, 60, 80, 100]
DEFAULT_N_EPOCHS = 20


def build_stiff_system(n_states):
    """Stiff system with constant lambda_max/lambda_min ratio."""
    magnitudes = np.geomspace(LAMBDA_MAX, LAMBDA_MAX / STIFFNESS_RATIO, n_states)
    A = np.diag(-magnitudes)
    B = np.ones((n_states, 1))
    Q = np.eye(n_states)
    R = 0.01 * np.eye(1)
    s0 = -np.ones(n_states)
    return A, B, Q, R, s0


def _create_layer(A, B, Q, R):
    """Bind the system matrices into the signature ``time_zoh_cell`` expects."""
    def factory(n_control, s0_np, n_s, n_u, u_max, x_max):
        return create_stiff_sys_zoh_clqr(
            n_control, s0_np, n_s, n_u, u_max, x_max,
            A_sys=A, B_sys=B, Q_sys=Q, R_sys=R,
        )
    return factory


def run(out_dir, n_epochs=DEFAULT_N_EPOCHS, n_states_list=None,
        n_control_list=None, backward_mode="lsqr"):
    n_states_list = n_states_list or N_STATES_LIST
    n_control_list = n_control_list or N_CONTROL_LIST

    A0, B0, Q0, R0, s00 = build_stiff_system(n_states_list[0])
    warmup(_create_layer(A0, B0, Q0, R0),
           A0, B0, Q0, R0, s00, T_HORIZON, U_MAX, X_MAX,
           backward_mode=backward_mode)

    rows = []
    pbar = tqdm(total=len(n_states_list) * len(n_control_list),
                desc="stiff_sys", unit="pt")
    for n_states in n_states_list:
        A, B, Q, R, s0 = build_stiff_system(n_states)
        factory = _create_layer(A, B, Q, R)
        for n_control in n_control_list:
            timing = time_zoh_cell(
                factory, A, B, Q, R, s0,
                T=T_HORIZON, n_control=n_control, n_epochs=n_epochs,
                u_max=U_MAX, x_max=X_MAX, backward_mode=backward_mode,
            )
            rows.append({"example": "stiff_sys", "n_states": n_states,
                         "n_control": n_control, "n_epochs": n_epochs,
                         **timing})
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "stiff_sys.csv")
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    out = make_run_dir(tag="stiff_sys")
    write_metadata(out)
    df = run(out)
    print(df.to_string(index=False))
    print(f"\nSaved to: {out}")
