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

## Directory Structure

Files are organized into example-specific subfolders. `utils.py`, `data/`, and `results/` remain shared at the top level.

### `pann_clqr/` — Pannocchia CLQR example
- **`pann_clqr.py`** — QP builders (CVXPY problem definitions + CvxpyLayer wrappers)
- **`pann_clqr_dt.py`** — Data generation: trains all 6 methods, saves pickles to `data/pann_clqr_dt/`
- **`plot_pann_clqr_dt.py`** / **`plot_pann_clqr_dt.ipynb`** — Visualization: loads pickles, produces plots in `results/pann_clqr_dt/`
- **`alt_losses.py`** — Data generation for alternative loss experiments, saves to `data/alt_losses/`
- **`plot_alt_losses.py`** / **`plot_alt_losses.ipynb`** — Visualization for alternative loss experiments

### `invpend/` — Inverted pendulum example
- **`invpend_clqr.py`** — QP builders for linearized cart-pole
- **`invpend_dt.py`** — Data generation: trains methods/losses, saves to `data/invpend_dt/`
- **`plot_invpend_dt.py`** / **`plot_invpend_dt.ipynb`** — Visualization: loads pickles, produces plots in `results/invpend_dt/`

### `stiff_sys/` — Stiff system LTI example
- **`stiff_sys_clqr.py`** — QP builders for diagonal system with time constants 0.1s, 10s, 100s
- **`stiff_sys_dt.py`** — Data generation: trains methods/losses, saves to `data/stiff_sys_dt/`
- **`plot_stiff_sys_dt.py`** — Visualization: loads pickles, produces plots in `results/stiff_sys_dt/`

### Shared (top level)
- **`utils.py`** — Shared utilities (discretization, loss functions, plotting, I/O)
- **`data/`** — Pickle files with saved results (subdirs per example)
- **`results/`** — Generated plots and figures (subdirs per example)

### Script Usage
```bash
# --- pann_clqr example ---
# Train all methods (5 test epochs)
python pann_clqr/pann_clqr_dt.py --mode test --method all

# Train specific methods (full run)
python pann_clqr/pann_clqr_dt.py --mode full --method rep zoh

# Generate all plots from saved data
python pann_clqr/plot_pann_clqr_dt.py

# Only cross-method analysis plots
python pann_clqr/plot_pann_clqr_dt.py --analysis-only

# Include baseline sweep
python pann_clqr/plot_pann_clqr_dt.py --baseline

# Alternative losses: train specific losses (test)
python pann_clqr/alt_losses.py --mode test --loss L_IV L_FI

# Alternative losses: train all (full run)
python pann_clqr/alt_losses.py --mode full --loss all

# Alternative losses: generate plots
python pann_clqr/plot_alt_losses.py

# Alternative losses: only analysis plots
python pann_clqr/plot_alt_losses.py --analysis-only

# --- invpend example ---
# Train all experiments (test)
python invpend/invpend_dt.py --mode test --experiment all

# Train specific method (full run)
python invpend/invpend_dt.py --mode full --experiment methods --method rep

# Generate all plots
python invpend/plot_invpend_dt.py

# --- stiff_sys example ---
# Train all experiments (test)
python stiff_sys/stiff_sys_dt.py --mode test --experiment all

# Train specific method (full run)
python stiff_sys/stiff_sys_dt.py --mode full --experiment methods --method rep

# Train specific losses (full run)
python stiff_sys/stiff_sys_dt.py --mode full --experiment losses --loss L_IV L_FI

# Generate all plots
python stiff_sys/plot_stiff_sys_dt.py

# Plot specific methods/losses
python stiff_sys/plot_stiff_sys_dt.py --method rep --loss L_IV L_FI

# Only cross-method analysis plots
python stiff_sys/plot_stiff_sys_dt.py --analysis-only

# Include baseline sweep
python stiff_sys/plot_stiff_sys_dt.py --baseline
```
