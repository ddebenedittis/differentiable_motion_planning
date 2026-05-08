"""Shared utilities for time-optimization benchmarks.

Provides:
    - ``time_zoh_cell``: time the dQP build + N Adam epochs through a CvxpyLayer
      for the exact-ZOH parametrization. Reused by every example so the
      numbers stay comparable.
    - ``write_metadata``: drop ``git_diff.txt`` and ``env.json`` (hostname,
      timestamp, commit, branch, python/torch versions) next to the CSVs so a
      future run can be matched back to the exact code state that produced it.
    - ``make_run_dir``: timestamped per-host output directory under
      ``benchmarks/time_optimization/results/``.
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Make the existing per-example QP builders importable from the notebooks tree.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NOTEBOOK_TIME_OPT = os.path.join(_REPO_ROOT, "notebooks", "time_optimization")
for _p in (_NOTEBOOK_TIME_OPT, os.path.join(_NOTEBOOK_TIME_OPT, "stiff_sys"),
           os.path.join(_NOTEBOOK_TIME_OPT, "pann")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import theta_2_dt, zoh_cost_matrices  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def make_run_dir(base=RESULTS_DIR, tag=None):
    """Create a fresh output directory: ``results/<timestamp>_<hostname>[_<tag>]``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().replace(" ", "_")
    name = f"{stamp}_{host}"
    if tag:
        name = f"{name}_{tag}"
    out_dir = os.path.join(base, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def write_metadata(out_dir, extra=None):
    """Persist hostname, timestamp, git commit/branch/diff into ``out_dir``."""
    diff = _git("diff", "HEAD")
    with open(os.path.join(out_dir, "git_diff.txt"), "w") as f:
        f.write(diff)

    env = {
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "git_dirty": bool(_git("status", "--porcelain").strip()),
    }
    if extra:
        env.update(extra)
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    return env


def time_zoh_cell(create_layer_fn, A_np, B_np, Q_np, R_np, s0_np, T, n_control,
                  n_epochs, u_max, x_max=None, backward_mode="lsqr",
                  dtype=torch.float32):
    """Time dQP creation + ``n_epochs`` Adam steps through the ZOH layer.

    ``create_layer_fn(n_control, s0_np, n_s, n_u, u_max, x_max)`` must return
    ``(_, layer, _, _, spec)`` matching the signature of
    ``create_exact_zoh_cost_clqr`` / ``create_stiff_sys_zoh_clqr``.

    Returns a dict with ``dQP_create``, ``train_total``, ``train_per_epoch``,
    ``total``.
    """
    n_s, n_u = A_np.shape[0], B_np.shape[1]
    A_t = torch.tensor(A_np, dtype=dtype)
    B_t = torch.tensor(B_np, dtype=dtype)
    Q_t = torch.tensor(Q_np, dtype=dtype)
    R_t = torch.tensor(R_np, dtype=dtype)
    s0_t = torch.tensor(s0_np, dtype=dtype)

    t0 = time.perf_counter()
    _, layer, _, _, spec = create_layer_fn(
        n_control, s0_np, n_s, n_u, u_max, x_max,
    )
    dQP_create = time.perf_counter() - t0

    theta = torch.nn.Parameter(torch.ones(n_control, 1, dtype=dtype))
    optim = torch.optim.Adam([theta], lr=1e-2)
    solver_args = {"mode": backward_mode, "solve_method": "CLARABEL"}

    t0 = time.perf_counter()
    for _ in range(n_epochs):
        optim.zero_grad(set_to_none=True)

        dts_torch = theta_2_dt(theta, T, n_control)
        packed_steps, W_list = [], []
        for k in range(n_control):
            Ad_k, Bd_k, W_k = zoh_cost_matrices(
                dts_torch[k], A_t, B_t, Q_t, R_t,
            )
            W_list.append(W_k)
            packed_steps.append(spec.pack_step(Ad_k, Bd_k, W_k))

        sol = layer(*spec.flatten_for_layer(packed_steps),
                    solver_args=solver_args)

        loss = torch.tensor(0.0, dtype=dtype)
        for k in range(n_control):
            s_k = s0_t if k == 0 else sol[k - 1].to(dtype)
            u_k = sol[n_control + k].to(dtype)
            z_k = torch.cat([s_k, u_k])
            loss = loss + z_k @ W_list[k] @ z_k

        loss.backward()
        optim.step()
    train_total = time.perf_counter() - t0

    return {
        "dQP_create": dQP_create,
        "train_total": train_total,
        "train_per_epoch": train_total / n_epochs,
        "total": dQP_create + train_total,
    }


def warmup(create_layer_fn, A_np, B_np, Q_np, R_np, s0_np, T, u_max,
           x_max=None, backward_mode="lsqr"):
    """Trigger first-call torch / CvxpyLayer overhead on a tiny problem."""
    time_zoh_cell(
        create_layer_fn, A_np, B_np, Q_np, R_np, s0_np,
        T=T, n_control=10, n_epochs=2, u_max=u_max, x_max=x_max,
        backward_mode=backward_mode,
    )
