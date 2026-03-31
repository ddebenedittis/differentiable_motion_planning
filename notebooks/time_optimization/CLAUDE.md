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
| **ZOH** | Exact integrated cost via Van Loan block matrix exponential; has `n10/n20/n40/n80` variants for convergence study | `sol_zoh`, `history_zoh`, `loss_methods_zoh` |

## Loss Functions
- **Unscaled**: $\sum_k (s_k^T Q s_k + u_k^T R u_k)$
- **Time-scaled**: $\sum_k \Delta t_k (s_k^T Q s_k + u_k^T R u_k)$ — approximates continuous integral

## Analysis Metrics
- **Sampling density**: $(1/\Delta t_k) \cdot (T/n)$, equals 1 for uniform sampling
- **Cross-correlation**: Time-lagged correlation between sampling density and $|\Delta u|$, $|\Delta s|$, $|u|$, $|s|$, $t$
- **Significance bounds**: $\pm 1.96/\sqrt{n}$ for 95% CI (values outside = statistically significant)

## Local Files
- **`pann_clqr_dt.py`** — Data generation script: trains all 6 methods, saves pickles to `data/pann_clqr_dt/`
- **`plot_pann_clqr_dt.py`** — Visualization script: loads pickles, produces plots in `results/pann_clqr_dt/`
- **`alt_losses.py`** — Data generation for alternative loss experiments (ZOH base + regularizer losses)
- **`plot_alt_losses.py`** — Visualization for alternative loss experiments (loads pickles, produces plots)
- **`pann_clqr.py`** — QP builders (CVXPY problem definitions + CvxpyLayer wrappers)
- **`utils.py`** — Shared utilities (discretization, loss functions, plotting, I/O)
- **`data/`** — Pickle files with saved results
- **`results/`** — Generated plots and figures

### Script Usage
```bash
# Train all methods (5 test epochs)
python pann_clqr_dt.py --mode test --method all

# Train specific methods (full run)
python pann_clqr_dt.py --mode full --method rep zoh

# Generate all plots from saved data
python plot_pann_clqr_dt.py

# Only cross-method analysis plots
python plot_pann_clqr_dt.py --analysis-only

# Include baseline sweep
python plot_pann_clqr_dt.py --baseline

# Alternative losses: train specific losses (test)
python alt_losses.py --mode test --loss L_IV L_FI

# Alternative losses: train all (full run)
python alt_losses.py --mode full --loss all

# Alternative losses: generate plots
python plot_alt_losses.py

# Alternative losses: only analysis plots
python plot_alt_losses.py --analysis-only
```
