#!/usr/bin/env python
"""Run all time-optimization speed benchmarks into one results folder.

Output layout::

    benchmarks/time_optimization/results/<timestamp>_<hostname>/
        stiff_sys.csv      stiff system (n_states x n_control sweep)
        pann.csv           Pannocchia (n_control sweep)
        all.csv            both, concatenated
        env.json           hostname, timestamp, git commit/branch, package versions
        git_diff.txt       output of `git diff HEAD` at run time

Usage::

    python benchmarks/time_optimization/run.py             # full sweep
    python benchmarks/time_optimization/run.py --epochs 5  # quick check
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import make_run_dir, write_metadata
import bench_pann
import bench_stiff_sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Adam epochs per cell (default: 20).")
    parser.add_argument("--backward-mode", default="lsqr",
                        choices=["dense", "lsqr"],
                        help="diffcp backward mode.")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory (default: timestamped).")
    args = parser.parse_args()

    out_dir = args.out_dir or make_run_dir()
    print(f"Output: {out_dir}\n")
    write_metadata(out_dir, extra={
        "n_epochs": args.epochs,
        "backward_mode": args.backward_mode,
    })

    df_stiff = bench_stiff_sys.run(
        out_dir, n_epochs=args.epochs, backward_mode=args.backward_mode,
    )
    df_pann = bench_pann.run(
        out_dir, n_epochs=args.epochs, backward_mode=args.backward_mode,
    )

    df_all = pd.concat([df_stiff, df_pann], ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "all.csv"), index=False)

    print("\n" + "=" * 80)
    print(df_all.to_string(index=False))
    print("=" * 80)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
