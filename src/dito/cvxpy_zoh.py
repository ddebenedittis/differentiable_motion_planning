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

    def make_step_params(self, k):
        """Build the dict of cp.Parameter objects for a single step k."""
        n_s, n_u, m = self.n_s, self.n_u, self.m
        if self.tier == "general":
            return {
                "Ad": cp.Parameter((n_s, n_s), name=f"Ad_{k}"),
                "Bd": cp.Parameter((n_s, n_u), name=f"Bd_{k}"),
                "Lx": cp.Parameter((m, n_s), name=f"Lx_{k}"),
                "Lu": cp.Parameter((m, n_u), name=f"Lu_{k}"),
            }
        if self.tier == "fully_diagonal":
            return {
                "Ad_diag": cp.Parameter(n_s, name=f"Adiag_{k}"),
                "Bd": cp.Parameter((n_s, n_u), name=f"Bd_{k}"),
                "L_diag": cp.Parameter(n_s, name=f"Ldiag_{k}"),
                "L_xu": cp.Parameter((n_s, n_u), name=f"Lxu_{k}"),
                "L_uu": cp.Parameter((n_u, n_u), name=f"Luu_{k}"),
            }
        # block_diagonal: at least one block has size > 1.
        Ad_blocks = []
        L11_blocks = []
        for bi, block in enumerate(self.blocks):
            b = len(block)
            if b == 1:
                Ad_blocks.append(cp.Parameter(1, name=f"Ad_{k}_b{bi}"))
                L11_blocks.append(cp.Parameter(1, name=f"L11_{k}_b{bi}"))
            else:
                Ad_blocks.append(cp.Parameter((b, b), name=f"Ad_{k}_b{bi}"))
                L11_blocks.append(cp.Parameter((b, b), name=f"L11_{k}_b{bi}"))
        return {
            "Ad_blocks": Ad_blocks,
            "L11_blocks": L11_blocks,
            "Bd": cp.Parameter((n_s, n_u), name=f"Bd_{k}"),
            "L_xu": cp.Parameter((n_s, n_u), name=f"Lxu_{k}"),
            "L_uu": cp.Parameter((n_u, n_u), name=f"Luu_{k}"),
        }

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

    def cost_expr(self, s_var, u_var, params):
        """cvxpy expression for `||L^T z||^2` for one step (z = [s; u]).

        Always emits exactly one `cp.sum_squares` atom per step, matching the
        legacy formulation's per-step cone count. Splitting into multiple
        `sum_squares` would double the SOC cone count in the diffcp affine
        map and slow the LSQR backward pass.
        """
        if self.tier == "general":
            return cp.sum_squares(params["Lx"] @ s_var + params["Lu"] @ u_var)

        if self.tier == "fully_diagonal":
            state_part = (
                cp.multiply(params["L_diag"], s_var) + params["L_xu"] @ u_var
            )
            input_part = params["L_uu"] @ u_var
            return cp.sum_squares(cp.hstack([state_part, input_part]))

        # block_diagonal: assemble the full length-m vector y = L^T @ z and
        # emit one sum_squares so the cone structure matches legacy exactly.
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
        return cp.sum_squares(cp.hstack([state_part, input_part]))

    def layer_parameters(self, step_params_list):
        """Flat list of cp.Parameters for `CvxpyLayer(parameters=...)`.

        Order must match `flatten_for_layer`.
        """
        if self.tier == "general":
            Ads = [p["Ad"] for p in step_params_list]
            Bds = [p["Bd"] for p in step_params_list]
            Lxs = [p["Lx"] for p in step_params_list]
            Lus = [p["Lu"] for p in step_params_list]
            return Ads + Bds + Lxs + Lus
        if self.tier == "fully_diagonal":
            Ads = [p["Ad_diag"] for p in step_params_list]
            Bds = [p["Bd"] for p in step_params_list]
            Lds = [p["L_diag"] for p in step_params_list]
            Lxus = [p["L_xu"] for p in step_params_list]
            Luus = [p["L_uu"] for p in step_params_list]
            return Ads + Bds + Lds + Lxus + Luus
        # block_diagonal
        n_blocks = len(self.blocks)
        out = []
        for bi in range(n_blocks):
            out.extend(p["Ad_blocks"][bi] for p in step_params_list)
        for bi in range(n_blocks):
            out.extend(p["L11_blocks"][bi] for p in step_params_list)
        out.extend(p["Bd"] for p in step_params_list)
        out.extend(p["L_xu"] for p in step_params_list)
        out.extend(p["L_uu"] for p in step_params_list)
        return out

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
        """Flat tuple of torch tensors for `layer(*tensors)`.

        Order matches `layer_parameters`.
        """
        if self.tier == "general":
            Ads = [p["Ad"] for p in packed_step_list]
            Bds = [p["Bd"] for p in packed_step_list]
            Lxs = [p["Lx"] for p in packed_step_list]
            Lus = [p["Lu"] for p in packed_step_list]
            return Ads + Bds + Lxs + Lus
        if self.tier == "fully_diagonal":
            Ads = [p["Ad_diag"] for p in packed_step_list]
            Bds = [p["Bd"] for p in packed_step_list]
            Lds = [p["L_diag"] for p in packed_step_list]
            Lxus = [p["L_xu"] for p in packed_step_list]
            Luus = [p["L_uu"] for p in packed_step_list]
            return Ads + Bds + Lds + Lxus + Luus
        # block_diagonal
        n_blocks = len(self.blocks)
        out = []
        for bi in range(n_blocks):
            out.extend(p["Ad_blocks"][bi] for p in packed_step_list)
        for bi in range(n_blocks):
            out.extend(p["L11_blocks"][bi] for p in packed_step_list)
        out.extend(p["Bd"] for p in packed_step_list)
        out.extend(p["L_xu"] for p in packed_step_list)
        out.extend(p["L_uu"] for p in packed_step_list)
        return out
