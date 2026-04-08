#!/usr/bin/env python3
"""
Four figures (English labels only), one experiment per pair of plots:

  plots/matmul_execution_time.png, plots/matmul_speedup.png
  plots/sin_taylor_execution_time.png, plots/sin_taylor_speedup.png

Times are hardcoded means of three runs from results/:

  matmul_bench_8000_0.txt .. matmul_bench_8000_2.txt
  sin_taylor_bench_0.txt, sin_taylor_bench_1.txt, sin_taylor_bench.txt

Thread order: 0 (serial), 2, 4, 8 OpenMP teams.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"

# ---------------------------------------------------------------------------
# Hardcoded mean wall time (seconds) after averaging three runs per benchmark
# ---------------------------------------------------------------------------
THREADS = (0, 2, 4, 8)

MATMUL_TIME_S = (
    831.4822312196667,
    432.224910577,
    229.88551163666667,
    132.01891970633334,
)

SIN_TIME_S = (
    226.89642206866666,
    116.28107865633334,
    61.30198001866666,
    32.35804494233333,
)


def speedup_vs_serial(times: tuple[float, ...]) -> tuple[float, ...]:
    t0 = times[0]
    return tuple(t0 / t for t in times)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(THREADS))
    xticks = [str(t) for t in THREADS]

    # --- Exp. 1: matrix multiply ---
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")
    ax.bar(x, MATMUL_TIME_S, width=0.55, color="#4C72B0")
    ax.set_xticks(x, xticks)
    ax.set_xlabel("Threads (0 = serial, else OpenMP team size)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Matrix multiply (N=8000): mean time (3 runs)")
    p1 = PLOTS / "matmul_execution_time.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    matmul_su = speedup_vs_serial(MATMUL_TIME_S)
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, zorder=0)
    ax.plot(THREADS, matmul_su, "o-", color="#4C72B0", markersize=7)
    ax.set_xticks(THREADS)
    ax.set_xlabel("Threads (0 = serial, else OpenMP team size)")
    ax.set_ylabel("Speedup vs. serial")
    ax.set_title("Matrix multiply (N=8000): speedup (3 runs)")
    p2 = PLOTS / "matmul_speedup.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # --- Exp. 2: sin Taylor ---
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")
    ax.bar(x, SIN_TIME_S, width=0.55, color="#DD8452")
    ax.set_xticks(x, xticks)
    ax.set_xlabel("Threads (0 = serial, else OpenMP team size)")
    ax.set_ylabel("Time (s)")
    ax.set_title(r"Sin Taylor ($x=1$, large $n$): mean time (3 runs)")
    p3 = PLOTS / "sin_taylor_execution_time.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    sin_su = speedup_vs_serial(SIN_TIME_S)
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, zorder=0)
    ax.plot(THREADS, sin_su, "s-", color="#DD8452", markersize=7)
    ax.set_xticks(THREADS)
    ax.set_xlabel("Threads (0 = serial, else OpenMP team size)")
    ax.set_ylabel("Speedup vs. serial")
    ax.set_title(r"Sin Taylor ($x=1$, large $n$): speedup (3 runs)")
    p4 = PLOTS / "sin_taylor_speedup.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    for path in (p1, p2, p3, p4):
        print(path)


if __name__ == "__main__":
    main()
