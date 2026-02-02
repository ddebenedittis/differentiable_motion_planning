# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DIMP (Differentiable Motion Planning) is a research project implementing differentiable trajectory optimization for robotic systems. It uses cvxpylayers to enable backpropagation through convex optimization problems, allowing end-to-end learning with motion planning.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip3 install hatchling
pip3 install -e .

# For notebooks
pip3 install -e ".[notebook]"

# Run tests
pytest

# Environment (required for Acados notebooks)
source .env
```

### Acados Setup (optional, for unicycle_acados.ipynb)
```bash
cd external && git clone https://github.com/acados/acados.git
cd acados && git submodule update --recursive --init
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON .. && make install -j4
cd ../../../ && pip install -e external/acados/interfaces/acados_template
```

## Architecture

### Robot Framework (`src/dimp/robots/`)

The core abstraction is a hierarchy for building MPC problems with CVXPY:

- **GeneralState/GeneralInput**: Base classes enforcing shape validation with named property accessors
- **MPCData**: Container managing states/inputs over a horizon with convenient indexing:
  - `mpc_data.statei[i]` - state at step i (cp.Variable or cp.Parameter)
  - `mpc_data.state` - all states stacked (cp.hstack)
  - `statesbar_list`/`inputsbar_list` - reference trajectories for linearization
- **GeneralRobot**: Abstract base defining `ct_dynamics()` (continuous-time) and `dt_dynamics_constraint()` (builds CVXPY constraints)

### Robot Models

| Model | State | Input | Notes |
|-------|-------|-------|-------|
| OmniRobot | [x, y] | [vx, vy] | Direct velocity control (holonomic) |
| PointMassRobot | [x, y, vx, vy] | [fx, fy] | Force-controlled, Newtonian dynamics |
| UniRobot | [x, y, theta] | [v, omega] | Linearized around `state_bar`/`input_bar` |

### Adding a New Robot

1. Subclass `GeneralState` with `n` (dimension) and `property_names`
2. Subclass `GeneralInput` similarly
3. Subclass `GeneralRobot` implementing `ct_dynamics(state, input, state_bar, input_bar)`

### Utilities (`src/dimp/utils/`)

- **disp.py**: `init_matplotlib()` for consistent styling, `get_colors()` for Okabe-Ito palette
- **disp_anim.py**: Animation framework for trajectory visualization
- **voronoi_task.py**: Bounded Voronoi diagrams for multi-robot task allocation

## Key Patterns

**MPC Problem Setup:**
```python
mpc_data = MPCData(
    nc=100,  # horizon
    states_list=[OmniState(s0)] + [OmniState(cp.Variable(2)) for _ in range(nc)],
    statesbar_list=[...],  # reference states (Parameters)
    inputs_list=[OmniInput(cp.Variable(2)) for _ in range(nc)],
    inputsbar_list=[...],  # reference inputs (Parameters)
)
robot = OmniRobot(dt=0.1, mpc_data=mpc_data)
constraints = robot.dt_dynamics_constraint()
```

**Nonlinear robots (UniRobot):** Require `update_bar()` after solving to update linearization point for iterative refinement.

## Notebooks

- `omni_go2goal_traj_opt.ipynb` - Basic trajectory optimization
- `omni_go2goal_layer.ipynb` - cvxpylayers differentiable optimization
- `mro_g2g.ipynb` - Multi-robot go-to-goal
- `notebooks/time_optimization/pann_clqr_dt.ipynb` - Differentiable time optimization (main reference)
- `unicycle_acados.ipynb` - Acados real-time optimal control

## Differentiable Time Optimization (`pann_clqr_dt.ipynb`)

This notebook explores **learning optimal discretization timesteps** for OCP via gradient descent through cvxpylayers.

### Problem Setup
- LTI system (Pannocchia example): 3 states, 1 input, T=10s horizon, n=160 timesteps
- Goal: Learn non-uniform Δtₖ that better approximates continuous-time optimal control

### Key Insight
Instead of uniform timesteps, make Δtₖ learnable parameters optimized to place **finer sampling where trajectory changes rapidly**.

### Approaches Implemented

| Method | Description | Key Variables |
|--------|-------------|---------------|
| **Aux (Auxiliary Variables)** | δₖ as QP optimization variables with regularization | `sol_aux`, `history_aux`, `loss_methods` |
| **Rep (Reparametrized)** | Softmax simplex: Δtₖ = ε + (T-nε)·softmax(θ)ₖ | `sol_rep`, `history_rep`, `loss_methods_2` |
| **HS (Hyper-Sampling)** | Evaluate loss on dense grid (uniform_resample or substeps) | `sol_hs`, `history_hs`, `loss_methods_hs` |
| **ZOH** | Exact discretization via matrix exponential | `sol_zoh`, `history_zoh`, `loss_methods_zoh` |
| **ZOH2** | ZOH with √Δt cost scaling in QP | `sol_zoh_2`, `history_zoh_2`, `loss_methods_zoh_2` |

### Loss Functions
- **Unscaled**: Σ(sₖᵀQsₖ + uₖᵀRuₖ)
- **Time-scaled**: Σ Δtₖ(sₖᵀQsₖ + uₖᵀRuₖ) — approximates continuous integral

### Analysis Metrics
- **Sampling density**: (1/Δtₖ)·(T/n), equals 1 for uniform sampling
- **Cross-correlation**: Time-lagged correlation between sampling density and |Δu|, |Δs|, |u|, |s|, t
- **Significance bounds**: ±1.96/√n for 95% CI (values outside = statistically significant)

### Data Storage
Results saved in `data/pann_clqr_dt/` as pickle files. Set `rerun=True` to regenerate.
