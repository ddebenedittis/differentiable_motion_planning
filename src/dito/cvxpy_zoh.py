"""Structure-aware ZOH parameter packing for cvxpylayers."""

import cvxpy as cp
import numpy as np
import torch


class ZOHParamSpec:
    """Structure-aware parameter declarations + packing for ZOH cvxpylayers.

    Detects block-diagonal structure of the joint A/Q sparsity graph (via
    union-find on edges where any of A_ij, A_ji, Q_ij, Q_ji is non-negligible)
    and dispatches to one of three tiers:

    - ``general``: A or Q couple all states. Uses dense legacy-shape `Lx (m, n_s)`
      and `Lu (m, n_u)` parameters, so cost is one `sum_squares(Lx @ s + Lu @ u)`
      atom per step — identical canonicalization to the legacy formulation.

    - ``fully_diagonal``: A and Q are jointly diagonal. The discrete-time
      `Ad`, `W_xx`, and Cholesky factor `L_11` are all diagonal, so each step
      stores `Ad_diag` and `L_diag` as 1-D parameters of length n_s. Per-step
      cost is two `sum_squares` atoms.

    - ``block_diagonal``: A and Q share a block-diagonal partition with at
      least one block of size > 1. Per block i: `Ad_i (b_i, b_i)` and the
      upper-triangular `L_11_i (b_i, b_i)` (lower-triangle entries are
      structurally zero from Cholesky). Per-step cost is one `sum_squares`
      per block plus one for `L_uu`.

    All tiers stay vectorized — at most O(n_blocks) atoms per step regardless
    of n_s — to keep cvxpy canonicalization fast at large state dimension.
    """

    def __init__(self, A, B, Q, R, *, atol=1e-12):
        A_np = np.asarray(A, dtype=float)
        B_np = np.asarray(B, dtype=float)
        Q_np = np.asarray(Q, dtype=float)
        R_np = np.asarray(R, dtype=float)
        n_s = A_np.shape[0]
        n_u = B_np.shape[1]
        if A_np.shape != (n_s, n_s):
            raise ValueError(f"A must be square, got {A_np.shape}")
        if Q_np.shape != (n_s, n_s):
            raise ValueError(f"Q must be (n_s, n_s), got {Q_np.shape}")
        if R_np.shape != (n_u, n_u):
            raise ValueError(f"R must be (n_u, n_u), got {R_np.shape}")

        self.n_s = n_s
        self.n_u = n_u
        self.m = n_s + n_u
        self.atol = atol

        # Sparsity-graph based block detection (joint A ∪ Q, undirected).
        A_inf = float(np.max(np.abs(A_np))) if n_s > 0 else 0.0
        Q_inf = float(np.max(np.abs(Q_np))) if n_s > 0 else 0.0
        thr_A = atol * (A_inf if A_inf > 0 else 1.0)
        thr_Q = atol * (Q_inf if Q_inf > 0 else 1.0)

        parent = list(range(n_s))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n_s):
            for j in range(i + 1, n_s):
                if (abs(A_np[i, j]) > thr_A or abs(A_np[j, i]) > thr_A
                        or abs(Q_np[i, j]) > thr_Q or abs(Q_np[j, i]) > thr_Q):
                    union(i, j)

        groups = {}
        for i in range(n_s):
            groups.setdefault(find(i), []).append(i)
        blocks = sorted((sorted(g) for g in groups.values()), key=lambda b: b[0])
        self.blocks = blocks
        self.is_block_structured = len(blocks) > 1
        self.is_fully_diagonal = (
            self.is_block_structured and all(len(b) == 1 for b in blocks)
        )
        self.block_indices_contiguous = all(
            b == list(range(b[0], b[0] + len(b))) for b in blocks
        )

        if not self.is_block_structured:
            self.tier = "general"
        elif self.is_fully_diagonal:
            self.tier = "fully_diagonal"
        else:
            self.tier = "block_diagonal"

        # Precomputed torch index buffers (CPU; safe to use across devices).
        self._block_idx_t = [torch.tensor(b, dtype=torch.long) for b in blocks]

    # ---------------------------------------------------------------- cvxpy side

    def make_step_params(self, n):
        """Build per-step view-dicts that slice into shared N-dim parameter stacks.

        Instead of declaring 5n separate `cp.Parameter` objects (one per
        timestep), this allocates a handful of stacked parameters of shape
        `(n, ...)` and exposes per-step `cp.Parameter[k]` views with the same
        legacy dict structure. Downstream expressions (`dynamics_expr`,
        `_cost_vector_expr`, `total_cost_expr`) work unchanged on the views.

        The stacks are stored on `self` so `layer_parameters` and
        `flatten_for_layer` can return / stack into them without re-deriving
        from the view-dicts.

        Args:
            n: horizon length (number of timesteps).

        Returns:
            list of length `n`; each entry is a dict with the same keys as
            the legacy per-step factory.
        """
        n_s, n_u, m = self.n_s, self.n_u, self.m
        self._n = n

        if self.tier == "general":
            self._Ad_all = cp.Parameter((n, n_s, n_s), name="Ad_all")
            self._Bd_all = cp.Parameter((n, n_s, n_u), name="Bd_all")
            self._Lx_all = cp.Parameter((n, m, n_s), name="Lx_all")
            self._Lu_all = cp.Parameter((n, m, n_u), name="Lu_all")
            return [
                {
                    "Ad": self._Ad_all[k],
                    "Bd": self._Bd_all[k],
                    "Lx": self._Lx_all[k],
                    "Lu": self._Lu_all[k],
                }
                for k in range(n)
            ]

        if self.tier == "fully_diagonal":
            self._Ad_diag_all = cp.Parameter((n, n_s), name="Ad_diag_all")
            self._Bd_all = cp.Parameter((n, n_s, n_u), name="Bd_all")
            self._L_diag_all = cp.Parameter((n, n_s), name="L_diag_all")
            self._L_xu_all = cp.Parameter((n, n_s, n_u), name="L_xu_all")
            self._L_uu_all = cp.Parameter((n, n_u, n_u), name="L_uu_all")
            return [
                {
                    "Ad_diag": self._Ad_diag_all[k],
                    "Bd": self._Bd_all[k],
                    "L_diag": self._L_diag_all[k],
                    "L_xu": self._L_xu_all[k],
                    "L_uu": self._L_uu_all[k],
                }
                for k in range(n)
            ]

        # block_diagonal: at least one block has size > 1.
        self._Ad_b_all = []
        self._L11_b_all = []
        for bi, block in enumerate(self.blocks):
            b = len(block)
            if b == 1:
                self._Ad_b_all.append(
                    cp.Parameter((n, 1), name=f"Ad_b{bi}_all")
                )
                self._L11_b_all.append(
                    cp.Parameter((n, 1), name=f"L11_b{bi}_all")
                )
            else:
                self._Ad_b_all.append(
                    cp.Parameter((n, b, b), name=f"Ad_b{bi}_all")
                )
                self._L11_b_all.append(
                    cp.Parameter((n, b, b), name=f"L11_b{bi}_all")
                )
        self._Bd_all = cp.Parameter((n, n_s, n_u), name="Bd_all")
        self._L_xu_all = cp.Parameter((n, n_s, n_u), name="L_xu_all")
        self._L_uu_all = cp.Parameter((n, n_u, n_u), name="L_uu_all")
        return [
            {
                "Ad_blocks": [
                    self._Ad_b_all[bi][k] for bi in range(len(self.blocks))
                ],
                "L11_blocks": [
                    self._L11_b_all[bi][k] for bi in range(len(self.blocks))
                ],
                "Bd": self._Bd_all[k],
                "L_xu": self._L_xu_all[k],
                "L_uu": self._L_uu_all[k],
            }
            for k in range(n)
        ]

    def dynamics_expr(self, s_var, u_var, params):
        """cvxpy expression for `Ad @ s + Bd @ u`."""
        if self.tier == "general":
            return params["Ad"] @ s_var + params["Bd"] @ u_var
        if self.tier == "fully_diagonal":
            return cp.multiply(params["Ad_diag"], s_var) + params["Bd"] @ u_var

        # block_diagonal
        Bd_term = params["Bd"] @ u_var  # (n_s,)
        if self.block_indices_contiguous:
            segments = []
            for block, Ad_b in zip(self.blocks, params["Ad_blocks"]):
                b = len(block)
                s_seg = s_var[block[0]:block[0] + b]
                if b == 1:
                    segments.append(cp.multiply(Ad_b, s_seg))
                else:
                    segments.append(Ad_b @ s_seg)
            stacked = segments[0] if len(segments) == 1 else cp.hstack(segments)
            return stacked + Bd_term

        # Non-contiguous fallback: place each block at its global indices.
        scalars = [None] * self.n_s
        for block, Ad_b in zip(self.blocks, params["Ad_blocks"]):
            b = len(block)
            if b == 1:
                s_seg = s_var[block[0]:block[0] + 1]
                seg = cp.multiply(Ad_b, s_seg)  # (1,)
                scalars[block[0]] = seg
            else:
                s_seg = cp.hstack([s_var[i] for i in block])
                seg = Ad_b @ s_seg  # (b,)
                for li, gi in enumerate(block):
                    scalars[gi] = seg[li:li + 1]
        return cp.hstack(scalars) + Bd_term

    def _cost_vector_expr(self, s_var, u_var, params):
        """Per-step cost vector y_k of length m such that step cost = ||y_k||^2.

        Returned separately from the sum_squares atom so that `total_cost_expr`
        can stack all steps and wrap them in a single sum_squares (one SOC
        cone for the whole horizon instead of n).
        """
        if self.tier == "general":
            return params["Lx"] @ s_var + params["Lu"] @ u_var

        if self.tier == "fully_diagonal":
            state_part = (
                cp.multiply(params["L_diag"], s_var) + params["L_xu"] @ u_var
            )
            input_part = params["L_uu"] @ u_var
            return cp.hstack([state_part, input_part])

        # block_diagonal: assemble the full length-m vector y = L^T @ z.
        L_xu = params["L_xu"]
        n_s = self.n_s

        if self.block_indices_contiguous:
            block_pieces = []
            for block, L11_b in zip(self.blocks, params["L11_blocks"]):
                b = len(block)
                s_block = s_var[block[0]:block[0] + b]
                Lxu_block = L_xu[block[0]:block[0] + b, :]
                if b == 1:
                    state_term = cp.multiply(L11_b, s_block)  # (1,)
                else:
                    state_term = L11_b @ s_block  # (b,)
                block_pieces.append(state_term + Lxu_block @ u_var)
            state_part = (
                block_pieces[0] if len(block_pieces) == 1
                else cp.hstack(block_pieces)
            )
        else:
            scalars = [None] * n_s
            for block, L11_b in zip(self.blocks, params["L11_blocks"]):
                b = len(block)
                if b == 1:
                    s_block = s_var[block[0]:block[0] + 1]
                    Lxu_block = L_xu[block[0]:block[0] + 1, :]
                    seg = cp.multiply(L11_b, s_block) + Lxu_block @ u_var  # (1,)
                    scalars[block[0]] = seg
                else:
                    s_block = cp.hstack([s_var[i] for i in block])
                    Lxu_block = L_xu[block, :]
                    seg = L11_b @ s_block + Lxu_block @ u_var  # (b,)
                    for li, gi in enumerate(block):
                        scalars[gi] = seg[li:li + 1]
            state_part = cp.hstack(scalars)

        input_part = params["L_uu"] @ u_var
        return cp.hstack([state_part, input_part])

    def cost_expr(self, s_var, u_var, params):
        """Per-step ``||L^T z||^2`` (z = [s; u]). One sum_squares per step.

        Kept for backward compatibility. New code should prefer
        :meth:`total_cost_expr`, which produces a single SOC cone across all
        timesteps instead of n separate ones.
        """
        return cp.sum_squares(self._cost_vector_expr(s_var, u_var, params))

    def total_cost_expr(self, s_vars, u_vars, step_params_list):
        """Single-SOC-cone cost over all timesteps.

        Math is identical to ``sum_k cost_expr(s_vars[k], u_vars[k],
        step_params_list[k])``: ``||hstack([L_k^T z_k for k])||^2`` equals
        ``sum_k ||L_k^T z_k||^2``. The difference is canonical form — this
        produces a single SOC cone (one ``format_constraints`` call) instead
        of n, which is the dominant ``CvxpyLayer.__init__`` cost.
        """
        pieces = [
            self._cost_vector_expr(s_var, u_var, params)
            for s_var, u_var, params in zip(
                s_vars, u_vars, step_params_list, strict=True
            )
        ]
        return cp.sum_squares(cp.hstack(pieces))

    def layer_parameters(self, step_params_list=None):
        """Flat list of stacked cp.Parameters for `CvxpyLayer(parameters=...)`.

        The stacks were allocated by `make_step_params(n)` and stored on
        `self`. The `step_params_list` argument is unused — kept only so
        existing callers (`spec.layer_parameters(step_params)`) continue to
        work without modification.

        Order must match `flatten_for_layer`.
        """
        if self.tier == "general":
            return [self._Ad_all, self._Bd_all, self._Lx_all, self._Lu_all]
        if self.tier == "fully_diagonal":
            return [
                self._Ad_diag_all, self._Bd_all,
                self._L_diag_all, self._L_xu_all, self._L_uu_all,
            ]
        # block_diagonal
        return (
            list(self._Ad_b_all) + list(self._L11_b_all)
            + [self._Bd_all, self._L_xu_all, self._L_uu_all]
        )

    # ---------------------------------------------------------------- torch side

    def pack_step(self, Ad, Bd, W):
        """Pack torch (Ad, Bd, W) into structured torch tensors per parameter."""
        n_s = self.n_s
        L = torch.linalg.cholesky(W)
        LT = L.T

        if self.tier == "general":
            return {
                "Ad": Ad,
                "Bd": Bd,
                "Lx": LT[:, :n_s],
                "Lu": LT[:, n_s:],
            }

        L_xu = LT[:n_s, n_s:]
        L_uu = LT[n_s:, n_s:]

        if self.tier == "fully_diagonal":
            return {
                "Ad_diag": torch.diagonal(Ad, 0),
                "Bd": Bd,
                "L_diag": torch.diagonal(LT[:n_s, :n_s], 0),
                "L_xu": L_xu,
                "L_uu": L_uu,
            }

        # block_diagonal: per-block (b, b) matrices. Cholesky gives an upper-tri
        # L^T, so the strictly-lower-tri entries of each LT11 block are exactly
        # zero — safe to store as a dense (b, b) parameter.
        LT11 = LT[:n_s, :n_s]
        Ad_blocks = []
        L11_blocks = []
        for block, idx_t in zip(self.blocks, self._block_idx_t):
            b = len(block)
            Ad_b = Ad.index_select(0, idx_t).index_select(1, idx_t)
            L11_b = LT11.index_select(0, idx_t).index_select(1, idx_t)
            if b == 1:
                Ad_blocks.append(Ad_b.reshape(1))
                L11_blocks.append(L11_b.reshape(1))
            else:
                Ad_blocks.append(Ad_b)
                L11_blocks.append(L11_b)
        return {
            "Ad_blocks": Ad_blocks,
            "L11_blocks": L11_blocks,
            "Bd": Bd,
            "L_xu": L_xu,
            "L_uu": L_uu,
        }

    def flatten_for_layer(self, packed_step_list):
        """Flat tuple of N-dim torch tensors for `layer(*tensors)`.

        Each parameter type is stacked once along dim 0 across the n
        timesteps. Order matches `layer_parameters`.
        """
        if self.tier == "general":
            return [
                torch.stack([p["Ad"] for p in packed_step_list], dim=0),
                torch.stack([p["Bd"] for p in packed_step_list], dim=0),
                torch.stack([p["Lx"] for p in packed_step_list], dim=0),
                torch.stack([p["Lu"] for p in packed_step_list], dim=0),
            ]
        if self.tier == "fully_diagonal":
            return [
                torch.stack([p["Ad_diag"] for p in packed_step_list], dim=0),
                torch.stack([p["Bd"] for p in packed_step_list], dim=0),
                torch.stack([p["L_diag"] for p in packed_step_list], dim=0),
                torch.stack([p["L_xu"] for p in packed_step_list], dim=0),
                torch.stack([p["L_uu"] for p in packed_step_list], dim=0),
            ]
        # block_diagonal
        n_blocks = len(self.blocks)
        out = []
        for bi in range(n_blocks):
            out.append(torch.stack(
                [p["Ad_blocks"][bi] for p in packed_step_list], dim=0,
            ))
        for bi in range(n_blocks):
            out.append(torch.stack(
                [p["L11_blocks"][bi] for p in packed_step_list], dim=0,
            ))
        out.append(torch.stack([p["Bd"] for p in packed_step_list], dim=0))
        out.append(torch.stack([p["L_xu"] for p in packed_step_list], dim=0))
        out.append(torch.stack([p["L_uu"] for p in packed_step_list], dim=0))
        return out
