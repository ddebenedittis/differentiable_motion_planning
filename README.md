# DIfferentiable Motion Planning (DIMP)

This is an equidistribution problem. You want each interval to carry equal "activity," meaning:

  $$\Delta t_k \cdot |\dot{u}_k| = \text{const} \quad \forall k$$

  which is equivalent to $\Delta t_k \propto 1/|\dot{u}_k|$ — exactly your requirement.

  The loss

  Penalize deviations from equidistribution:

  $$L = \sum_k \left( \Delta t_k \cdot |\dot{u}_k| ;-; \bar{m} \right)^2, \qquad \bar{m} = \frac{1}{n}\sum_j \Delta t_j \cdot |\dot{u}_j|$$

  In discrete terms, $\dot{u}k \approx \frac{u{k+1} - u_k}{\Delta t_k}$, so $\Delta t_k \cdot |\dot{u}_k| \approx |\Delta u_k|$, and the loss becomes:

  $$L = \operatorname{Var}!\bigl(|\Delta u_k|\bigr) = \sum_k \left( |\Delta u_k| - \overline{|\Delta u|} \right)^2$$

  This looks like it doesn't involve $\Delta t_k$ explicitly, but the gradients flow through the OCP solution ($u_k$ depends on the discretization).

  Why this works

  - If an interval has large $|\dot{u}_k|$ and large $\Delta t_k$: its activity exceeds the mean → loss pushes $\Delta t_k$ down (denser sampling)
  - If an interval has small $|\dot{u}_k|$ and small $\Delta t_k$: its activity is below the mean → loss pushes $\Delta t_k$ up (coarser sampling)
  - The constraint $\sum \Delta t_k = T$ redistributes the total budget

  Alternative: weighted integral

  A simpler (but less principled) option:

  $$L = \sum_k \Delta t_k \cdot |\dot{u}_k|^2$$

  This approximates $\int |\dot{u}|^2 , dt$ on the non-uniform grid. Minimizing over the grid placement naturally concentrates points where $|\dot{u}|$ is large. However, being linear in $\Delta t_k$, it can produce bang-bang
  solutions unless combined with a regularizer.

  The variance-based equidistribution loss is more robust and directly encodes the proportionality you want.

## Installation

Create a [Python virtual environment](https://docs.python.org/3/library/venv.html) or whatever you enjoy.
If you do not use a virtual environment named `.venv`, you will need to change the `.vscode/settings.json` file accordingly.

Create the virtual environment with:
```bash
python3 -m venv .venv
```
Activate the virtual environment (on Ubuntu) with:
```bash
source .venv/bin/activate
```

Install the package with:
```bash
pip3 install hatchling
pip3 install -e .
```

### Acados

Following [this](https://docs.acados.org/python_interface/index.html), do
```bash
mkdir -p external
cd external
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
# add more optional arguments e.g. -DACADOS_WITH_DAQP=ON, a list of CMake options is provided below
make install -j4
```

Go back to the root folder and install acados with
```bash
pip install -e external/acados/interfaces/acados_template
```

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$PWD/external/acados/lib"
export ACADOS_SOURCE_DIR="$PWD/external/acados"
```

## Usage

So empty...

## Development

### Pre-commit hooks

Pre commit hooks are used to clean up Jupyter notebooks.

Install pre-commit with:
```bash
pip3 install pre-commit
```

To install pre-commit hooks, run:
```bash
pre-commit install
```

## Notebooks

After trying out [Marimo](https://marimo.io/), I decided to go back to Jupyter notebooks.
Sad.

- [`pann_clqr_dt.ipynb`](notebooks/time_optimization/pann_clqr_dt.ipynb): solves the CLQR problem from ["Whither discrete time model predictive control?"](https://doi.org/10.1109/TAC.2014.2324131) and "Optimal Non-Uniform Time Sampling in Continuous-Time Constrained LQR" using differentiable convex optimization. Takes 30ish minutes to run.

## Troubleshooting

