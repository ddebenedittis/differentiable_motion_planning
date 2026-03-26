# CLAUDE.md

## Project Overview

DIMP (Differentiable Motion Planning) is a research project implementing differentiable trajectory optimization for robotic systems. It uses cvxpylayers to enable backpropagation through convex optimization problems, allowing end-to-end learning with motion planning.

Key dependencies: `cvxpy`, `cvxpylayers`, `torch`, `matplotlib`, `scipy`.

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

### Imports / Public API

```python
from dimp.robots import (
    GeneralState, GeneralInput, GeneralRobot, MPCData,
    ForceInput,
    OmniState, OmniInput, OmniRobot,
    PointMassState, PointMassInput, PointMassRobot,
    UniState, UniInput, UniRobot,
)
from dimp.utils import (
    init_matplotlib, get_colors, gen_arrow_head_marker,
    Animation, Player, MultiRobotArtistFlags,
    display_animation, save_snapshots, plot_distances, plot_colour_line,
)
```

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

## Conventions

- **Pre-commit hook**: `nb-clean` strips notebook metadata/empty cells on commit
- **Data persistence**: Pickle files in `data/` dirs; notebooks use `rerun=True` flag to skip re-computation
- **Visualization**: Always call `init_matplotlib()` before plotting

## Notebooks

- `gain_optimization/omni_go2goal_traj_opt.ipynb` — Basic trajectory optimization
- `gain_optimization/omni_go2goal_layer.ipynb` — cvxpylayers differentiable optimization
- `gain_optimization/unicycle_go2goal.ipynb` — Unicycle trajectory optimization with SQP
- `gain_optimization/point_mass_push.ipynb` — Point mass with obstacle/contact constraints
- `gain_optimization/n_ode.ipynb` — Neural ODE learning of unicycle dynamics
- `gain_optimization/mro_g2g.ipynb` — Multi-robot go-to-goal
- `gain_optimization/unicycle_acados.ipynb` — Acados real-time optimal control
- `time_optimization/pann_clqr_dt.py` — Data generation script (training + pickle saving)
- `time_optimization/plot_pann_clqr_dt.py` — Visualization script (loads pickles, produces plots)
- `time_optimization/run_alt_losses.py` — Alternative loss function experiments

