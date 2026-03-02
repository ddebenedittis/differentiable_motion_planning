# Differentiable Time Optimization

This directory explores **learning optimal discretization timesteps** for OCP via gradient descent through cvxpylayers.

## Problem Setup
- LTI system (Pannocchia example): 3 states, 1 input, T=10s horizon, n=160 timesteps
- Goal: Learn non-uniform $\Delta t_k$ that better approximates continuous-time optimal control

## Key Insight
Instead of uniform timesteps, make $\Delta t_k$ learnable parameters optimized to place **finer sampling where trajectory changes rapidly**.

## Approaches Implemented

| Method | Description | Key Variables |
|--------|-------------|---------------|
| **Aux (Auxiliary Variables)** | $\delta_k$ as QP optimization variables with regularization | `sol_aux`, `history_aux`, `loss_methods` |
| **Rep (Reparametrized)** | Softmax simplex: $\Delta t_k = \varepsilon + (T - n\varepsilon) \cdot \text{softmax}(\theta)_k$ | `sol_rep`, `history_rep`, `loss_methods_2` |
| **HS (Hyper-Sampling)** | Evaluate loss on dense grid (uniform_resample or substeps) | `sol_hs`, `history_hs`, `loss_methods_hs` |
| **ZOH** | Exact discretization via matrix exponential | `sol_zoh`, `history_zoh`, `loss_methods_zoh` |
| **ZOH2** | ZOH with $\sqrt{\Delta t}$ cost scaling in QP | `sol_zoh_2`, `history_zoh_2`, `loss_methods_zoh_2` |
| **ZOH3** | Exact integrated cost via Van Loan block matrix exponential; has `n10/n20/n40/n80` variants for convergence study | `sol_zoh_3`, `history_zoh_3`, `loss_methods_zoh_3` |

## Loss Functions
- **Unscaled**: $\sum_k (s_k^T Q s_k + u_k^T R u_k)$
- **Time-scaled**: $\sum_k \Delta t_k (s_k^T Q s_k + u_k^T R u_k)$ — approximates continuous integral

## Analysis Metrics
- **Sampling density**: $(1/\Delta t_k) \cdot (T/n)$, equals 1 for uniform sampling
- **Cross-correlation**: Time-lagged correlation between sampling density and $|\Delta u|$, $|\Delta s|$, $|u|$, $|s|$, $t$
- **Significance bounds**: $\pm 1.96/\sqrt{n}$ for 95% CI (values outside = statistically significant)

## Local Files
- **`pann_clqr_dt.ipynb`** — Main notebook with all experiments
- **`pann_clqr.py`** — Problem definition and system matrices
- **`utils.py`** — Shared utilities for this topic (resampling, plotting helpers)
- **`data/`** — Pickle files with saved results. Set `rerun=True` in notebook to regenerate.
