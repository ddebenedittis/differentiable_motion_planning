#!/bin/bash
# Multi-sample wrapper for bench_creation.py. Runs N times and copies the
# output to creation_breakdown__post_A_<suffix>_runK.csv (K=1..N).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi
SUFFIX="$1"; N="$2"; shift 2
for ((i=1; i<=N; i++)); do
    echo "=== Sample $i/$N ===" >&2
    python notebooks/time_optimization/bench_creation.py "$@" 2>&1 | tail -2
    cp notebooks/time_optimization/bench/creation_breakdown.csv \
       "notebooks/time_optimization/bench/creation_breakdown__post_A_${SUFFIX}_run${i}.csv"
done
