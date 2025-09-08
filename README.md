# DIfferentiable Motion Planning (DIMP)

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

