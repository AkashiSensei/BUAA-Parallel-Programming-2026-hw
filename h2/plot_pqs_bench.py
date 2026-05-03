#!/usr/bin/env python3
"""
Hard-coded plotting script for the pthread parallel quicksort benchmark.
No CLI args: edit the arrays below if you re-run experiments.

Outputs (English labels) to ./plots/
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    # Average wall time (seconds) from bench output (5 runs each).
    threads = [1, 2, 3, 4, 6, 8]
    time_sec = [18.212417, 10.134970, 7.242837, 5.660566, 4.113904, 3.347781]

    t1 = time_sec[0]
    speedup = [t1 / t for t in time_sec]

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Time vs threads
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(threads, time_sec, marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Time (s)")
    plt.title("Parallel Quicksort: Time vs Threads")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xticks(threads)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pqs_time_vs_threads.png"), dpi=200)
    plt.close()

    # Speedup vs threads (vs 1 thread)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(threads, speedup, marker="o", color="tab:orange")
    plt.xlabel("Threads")
    plt.ylabel("Speedup (vs 1 thread)")
    plt.title("Parallel Quicksort: Speedup vs Threads")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xticks(threads)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pqs_speedup_vs_threads.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
