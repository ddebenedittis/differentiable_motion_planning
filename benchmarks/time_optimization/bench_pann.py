"""Pannocchia speed benchmark: n_control in {50, 100}.

Uses the exact-ZOH parametrization (``create_exact_zoh_cost_clqr``) — same
codepath the training script in ``notebooks/time_optimization/pann`` exercises.
"""

import os

import pandas as pd
from tqdm import tqdm

from _common import make_run_dir, time_zoh_cell, warmup, write_metadata
from pann_prob import (  # noqa: E402
    A, B, Q, R, T, create_exact_zoh_cost_clqr, s0, u_max,
)

N_CONTROL_LIST = [50, 100]
DEFAULT_N_EPOCHS = 20


def _factory(n_control, s0_np, n_s, n_u, _u_max, _x_max):
    """Adapter so ``time_zoh_cell`` can call the Pannocchia builder uniformly."""
    return create_exact_zoh_cost_clqr(n_control, s0_np, n_s, n_u, u_max=_u_max)


def run(out_dir, n_epochs=DEFAULT_N_EPOCHS, n_control_list=None,
        backward_mode="lsqr"):
    n_control_list = n_control_list or N_CONTROL_LIST

    warmup(_factory, A, B, Q, R, s0, T, u_max, x_max=None,
           backward_mode=backward_mode)

    rows = []
    pbar = tqdm(total=len(n_control_list), desc="pann", unit="pt")
    for n_control in n_control_list:
        timing = time_zoh_cell(
            _factory, A, B, Q, R, s0,
            T=T, n_control=n_control, n_epochs=n_epochs,
            u_max=u_max, x_max=None, backward_mode=backward_mode,
        )
        rows.append({"example": "pann", "n_states": A.shape[0],
                     "n_control": n_control, "n_epochs": n_epochs,
                     **timing})
        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "pann.csv")
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    out = make_run_dir(tag="pann")
    write_metadata(out)
    df = run(out)
    print(df.to_string(index=False))
    print(f"\nSaved to: {out}")
