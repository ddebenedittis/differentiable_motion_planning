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

## Directory Structure

Files are organized into example-specific subfolders. `utils.py`, `data/`, and `results/` remain shared at the top level.

### File Naming Convention

Each example `<exp>` lives in its own subfolder and contains three standard scripts:

| File | Role |
|------|------|
| `<exp>_prob.py` | QP builders + system constants (A, B, s0, T, Q, R, ...) |
| `<exp>_train.py` | Data generation: trains methods/losses, saves pickles to `data/` |
| `<exp>_plot.py` (+ `.ipynb`) | Visualization: loads pickles, produces plots in `results/` |

System constants are defined once in `*_prob.py` and imported by training/plotting scripts.

### Examples

- **`pann/`** — Pannocchia CLQR (3 states, 1 input)
- **`invpend/`** — Inverted pendulum (linearized cart-pole, error coordinates)
- **`stiff_sys/`** — Stiff system LTI (well-separated time constants: 0.1s, 10s, 100s)

### Shared (top level)
- **`utils.py`** — Shared utilities (discretization, loss functions, plotting helpers, I/O)
- **`losses_config.json`** — Shared config controlling which aux losses run and their `lambda0` (see below)
- **`data/`** — Pickle files with saved results (subdirs per example)
- **`results/`** — Generated plots and figures (subdirs per example)

### Losses Config (`losses_config.json`)

Controls which aux losses run across **all** scenarios when `--loss all` is passed.
Edit this one file to add/remove losses globally.

```json
{
  "enabled": ["L_IV", "L_FI", "L_PWLH", "L_SSD"],
  "per_loss": {
    "L_IV":   {"lambda0": 0.3},
    "L_FI":   {"lambda0": 0.5}
  }
}
```

- `enabled` — the set of losses run when `--loss all` is used (replaces the full `LOSS_REGISTRY` default)
- `per_loss` — optional per-loss `lambda0` override (falls back to `--lambda0` CLI arg)
- `--loss L_IV L_FI` always overrides the config (explicit list ignores `enabled`)
- `--config PATH` points to a custom config file

### Script Usage
```bash
# Generic pattern (replace <dir>/<exp> with e.g. pann/pann, invpend/invpend, stiff_sys/stiff_sys):
python <dir>/<exp>_train.py --mode test --experiment all
python <dir>/<exp>_train.py --mode full --experiment methods --method rep
python <dir>/<exp>_train.py --mode full --experiment losses --loss L_IV L_FI
python <dir>/<exp>_train.py --mode full --experiment losses  # uses losses_config.json
python <dir>/<exp>_plot.py
python <dir>/<exp>_plot.py --method rep --loss L_IV L_FI
python <dir>/<exp>_plot.py --analysis-only
python <dir>/<exp>_plot.py --baseline
```
