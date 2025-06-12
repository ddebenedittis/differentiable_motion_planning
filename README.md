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

## Usage

So empty...

## Notebooks

All the notebooks do use [Marimo](https://marimo.io/), instead of classic Jupyter notebooks.
Look it up 🙄.
There is also a VS Code extension for it.

## Troubleshooting

Marimo notebooks sucks a bit and get stuck sometimes.
In that case, use:
```bash
pkill -f marimo
```