"""R-adaptive merge/split mesh moves with Metropolis acceptance."""

import math

import numpy as np
import torch


def compute_importance(states, inputs, dts, Q, R, mode="cost_density"):
    """Per-timestep and per-pair importance scores for r-adaptive merge/split.

    Returns (eta_single, eta_pair):
        eta_single: shape (n,) — per-timestep score for split selection.
        eta_pair:   shape (n-1,) — per-adjacent-pair score for merge selection.

    Both arrays use averaging (not summing) over their constituents so that
    boundary timesteps with fewer neighbors are not down-weighted.

    Modes:
        "cost_density":
            single[k] = s_k'Q s_k + u_k'R u_k
            pair[k]   = (single[k] + single[k+1]) / 2
        "control_var":
            Let du2[k] = ||u_{k+1} - u_k||^2 for k in [0, n-2].
            single[k] = mean of available adjacent du2:
                interior:    (du2[k-1] + du2[k]) / 2
                edge k=0:    du2[0]
                edge k=n-1:  du2[n-2]
            pair[k]   = du2[k]
        "combined":
            geometric mean of cost_density and control_var, applied separately
            to single and pair arrays.
    """
    n = len(inputs)

    if mode == "cost_density":
        Q_t = torch.as_tensor(Q, dtype=dts.dtype, device=dts.device)
        R_t = torch.as_tensor(R, dtype=dts.dtype, device=dts.device)
        eta_t = torch.zeros(n, dtype=dts.dtype, device=dts.device)
        for k in range(n):
            s_k = states[k]
            u_k = inputs[k]
            eta_t[k] = s_k @ Q_t @ s_k + u_k @ R_t @ u_k
        eta_single = eta_t.detach().cpu().numpy()
        eta_pair = 0.5 * (eta_single[:-1] + eta_single[1:])
        return eta_single, eta_pair

    if mode == "control_var":
        if n < 2:
            return np.zeros(n), np.zeros(max(n - 1, 0))
        du2_t = torch.zeros(n - 1, dtype=dts.dtype, device=dts.device)
        for k in range(n - 1):
            du = inputs[k + 1] - inputs[k]
            du2_t[k] = torch.sum(du * du)
        du2 = du2_t.detach().cpu().numpy()
        eta_single = np.empty(n, dtype=du2.dtype)
        eta_single[0] = du2[0]
        eta_single[n - 1] = du2[n - 2]
        if n > 2:
            eta_single[1:-1] = 0.5 * (du2[:-1] + du2[1:])
        eta_pair = du2
        return eta_single, eta_pair

    if mode == "combined":
        cd_single, cd_pair = compute_importance(
            states, inputs, dts, Q, R, "cost_density")
        cv_single, cv_pair = compute_importance(
            states, inputs, dts, Q, R, "control_var")
        eta_single = np.sqrt(
            np.clip(cd_single, 0.0, None) * np.clip(cv_single, 0.0, None))
        eta_pair = np.sqrt(
            np.clip(cd_pair, 0.0, None) * np.clip(cv_pair, 0.0, None))
        return eta_single, eta_pair

    raise ValueError(f"Unknown importance mode: {mode}")


def select_merge_split(eta_single, eta_pair, theta_np, beta=1.0, rng=None):
    """Select an adjacent merge pair (j, j+1) and a split index i.

    With finite `beta`, sampled by softmax: merge ~ exp(-beta * eta_pair),
    split ~ exp(beta * eta_single). With `beta = np.inf` the selection is
    deterministic: j = argmin(eta_pair), i = argmax(eta_single) over indices
    outside {j, j+1}.

    Returns (j, i) or None when n < 3.
    """
    n = len(eta_single)
    if n < 3:
        return None

    eta_pair_arr = np.asarray(eta_pair)
    eta_single_arr = np.asarray(eta_single)

    if np.isposinf(beta):
        j = int(np.argmin(eta_pair_arr))
        valid_mask = np.ones(n, dtype=bool)
        valid_mask[j] = False
        valid_mask[j + 1] = False
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return None
        i = int(valid_idx[np.argmax(eta_single_arr[valid_idx])])
        return j, i

    if rng is None:
        rng = np.random.default_rng()

    pair_logits = -beta * eta_pair_arr
    pair_logits = pair_logits - pair_logits.max()
    pair_w = np.exp(pair_logits)
    pair_p = pair_w / pair_w.sum()
    j = int(rng.choice(n - 1, p=pair_p))

    valid = np.array([k for k in range(n) if k != j and k != j + 1])
    if len(valid) == 0:
        return None
    split_logits = beta * eta_single_arr[valid]
    split_logits = split_logits - split_logits.max()
    split_w = np.exp(split_logits)
    split_p = split_w / split_w.sum()
    i = int(rng.choice(valid, p=split_p))

    return j, i


def apply_merge_split(theta_np, j, i):
    """Apply a merge-split move in theta-space (softmax-mass preserving).

    Merge (j, j+1):  theta_new[j] = logsumexp(theta_j, theta_{j+1}); drop j+1.
    Split i (post-merge index): replace by two entries each = theta_i - log 2.

    The total softmax mass is preserved exactly (sum stays 1, dt on simplex).

    Args:
        theta_np: numpy array of shape (n, 1).
        j: merge index (0..n-2).
        i: split index in original numbering, must satisfy i not in {j, j+1}.

    Returns:
        new_theta: numpy array of shape (n, 1).
    """
    if i == j or i == j + 1:
        raise ValueError(
            f"split index i={i} cannot be in merge pair (j={j}, j+1={j+1})")

    n = theta_np.shape[0]
    theta_flat = theta_np.flatten().copy()
    dtype = theta_flat.dtype

    a, b = theta_flat[j], theta_flat[j + 1]
    M = max(a, b)
    theta_merged = M + np.log(np.exp(a - M) + np.exp(b - M))

    merged = np.delete(theta_flat, j + 1)
    merged[j] = theta_merged

    i_new = i - 1 if i > j + 1 else i

    split_val = merged[i_new] - np.array(np.log(2.0), dtype=dtype)

    new_flat = np.empty(n, dtype=dtype)
    new_flat[:i_new] = merged[:i_new]
    new_flat[i_new] = split_val
    new_flat[i_new + 1] = split_val
    new_flat[i_new + 2:] = merged[i_new + 1:]

    return new_flat.reshape(n, 1)


class RAdaptDriver:
    """Driver for r-adaptive merge/split moves with Metropolis acceptance.

    Encapsulates the frequency/temperature schedules, RNG, and the
    propose/apply/accept-or-reject cycle used inside training loops.
    Each epoch the training loop calls `maybe_step(...)`, which returns a
    dict of `radapt_*` fields ready to merge into a history entry.

    The driver requires an `evaluate_fn(epoch) -> dict` callable that, when
    invoked, returns at least `{"loss", "states", "inputs", "dts"}` from a
    forward pass at the current `theta`.
    """

    NULL_FIELDS = {
        "radapt_attempted": False, "radapt_accepted": None,
        "radapt_j": None, "radapt_i": None,
        "radapt_dL": None, "radapt_T": None,
    }

    def __init__(self, *, n_epochs, Q, R,
                 enable=False, every=10, freq_schedule=None,
                 importance="combined", beta=1.0,
                 temp_schedule=("exp", 1.0, 0.01),
                 warmup=20, tail_skip=20, accept_cooldown=0, seed=None):
        self.enable = bool(enable)
        self.n_epochs = int(n_epochs)
        self.Q = Q
        self.R = R
        self.every = int(every)
        self.importance = importance
        self.beta = float(beta)
        self.warmup = int(warmup)
        self.tail_skip = int(tail_skip)
        self.accept_cooldown = int(accept_cooldown)
        self.rng = np.random.default_rng(seed)
        self._last_accept_epoch = None

        self._temp_at = self._build_temp_schedule(temp_schedule)
        self._freq_at = self._build_freq_schedule(freq_schedule)

    def _build_temp_schedule(self, temp_schedule):
        n_epochs = self.n_epochs
        if temp_schedule is None:
            def temp_at(epoch):
                return 0.0
            return temp_at

        kind, T_start, T_end = temp_schedule
        if kind not in ("linear", "exp"):
            raise ValueError(
                f"radapt temp_schedule kind must be 'linear' or 'exp', got '{kind}'")
        if T_start <= 0 or T_end <= 0:
            raise ValueError("radapt temperature endpoints must be positive")
        if kind == "linear":
            def temp_at(epoch):
                if n_epochs <= 1:
                    return float(T_end)
                t = epoch / (n_epochs - 1)
                return float(T_start + (T_end - T_start) * t)
        else:
            log_start = np.log(T_start)
            log_end = np.log(T_end)
            def temp_at(epoch):
                if n_epochs <= 1:
                    return float(T_end)
                t = epoch / (n_epochs - 1)
                return float(np.exp(log_start + (log_end - log_start) * t))
        return temp_at

    def _build_freq_schedule(self, freq_schedule):
        if freq_schedule is None:
            every = self.every
            def freq_at(epoch):
                return int(every)
            return freq_at

        f_start, f_end = freq_schedule
        active_start = self.warmup
        active_end = max(active_start, self.n_epochs - self.tail_skip - 1)
        def freq_at(epoch):
            span = active_end - active_start
            if span <= 0:
                return max(1, int(f_end))
            t = (epoch - active_start) / span
            t = max(0.0, min(1.0, t))
            return max(1, int(round(f_start + (f_end - f_start) * t)))
        return freq_at

    def _is_due(self, epoch):
        if not self.enable:
            return False
        if not (self.warmup <= epoch < self.n_epochs - self.tail_skip):
            return False
        if (self._last_accept_epoch is not None
                and epoch - self._last_accept_epoch <= self.accept_cooldown):
            return False
        every_now = self._freq_at(epoch)
        if every_now <= 0:
            return False
        return (epoch - self.warmup) % every_now == 0

    def maybe_step(self, epoch, theta, evaluate_fn, optim):
        """If a move is due this epoch, attempt it and update `theta` in place.

        Args:
            epoch: current epoch index (0-based).
            theta: torch.nn.Parameter holding softmax logits, shape (n, 1).
            evaluate_fn: callable mapping epoch -> dict containing at least
                {"loss", "states", "inputs", "dts"}.
            optim: torch optimizer holding `theta`. Its state is cleared on
                accept so stale momentum does not pull theta back.

        Returns:
            dict of `radapt_*` fields to merge into the history entry.
        """
        if not self._is_due(epoch):
            return dict(self.NULL_FIELDS)

        theta_snapshot = theta.detach().clone()
        with torch.no_grad():
            out_post = evaluate_fn(epoch)
        loss_before = float(out_post["loss"].item())

        eta_single, eta_pair = compute_importance(
            out_post["states"], out_post["inputs"], out_post["dts"],
            self.Q, self.R, mode=self.importance,
        )
        sel = select_merge_split(
            eta_single, eta_pair, theta.detach().cpu().numpy(),
            beta=self.beta, rng=self.rng,
        )
        if sel is None:
            return dict(self.NULL_FIELDS)

        j, i = sel
        theta_new_np = apply_merge_split(
            theta.detach().cpu().numpy(), j, i,
        )
        with torch.no_grad():
            theta.copy_(torch.tensor(theta_new_np, dtype=theta.dtype))
            out_after = evaluate_fn(epoch)
        loss_after = float(out_after["loss"].item())

        T_metro = self._temp_at(epoch)
        dL = loss_after - loss_before
        if dL <= 0:
            accept = True
        elif T_metro <= 1e-12:
            accept = False
        else:
            accept = self.rng.random() < math.exp(-dL / T_metro)

        if accept:
            optim.state.clear()
            self._last_accept_epoch = epoch
        else:
            with torch.no_grad():
                theta.copy_(theta_snapshot)

        return {
            "radapt_attempted": True,
            "radapt_accepted": bool(accept),
            "radapt_j": int(j),
            "radapt_i": int(i),
            "radapt_dL": float(dL),
            "radapt_T": float(T_metro),
        }
