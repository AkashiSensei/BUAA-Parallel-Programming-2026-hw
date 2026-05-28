#!/usr/bin/env python3
"""Plot benchmark time and speedup from results/matmul_bench.csv (five-rep distribution)."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLOR_TIME = "#4e79a7"
COLOR_SPEEDUP = "#59a14f"
COLOR_IDEAL = "#bab0ac"
COLOR_ANOMALY = "#e15759"
BOX_EDGE = "#4a5568"
BOX_FILL_TIME = "#c8d9eb"
BOX_FILL_SPEEDUP = "#c9e4ca"

FIG_BG = "#FFFFFF"
GRID_COLOR = "#DDE4EA"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


def load_bench(
    csv_path: Path, n: int
) -> tuple[list[int], dict[int, list[float]], dict[int, tuple[int, int]]]:
    by_p_rep: dict[int, dict[int, float]] = defaultdict(dict)
    grid: dict[int, tuple[int, int]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["N"]) != n:
                continue
            p = int(row["p"])
            rep = int(row["rep"])
            by_p_rep[p][rep] = float(row["time_sec"])
            grid[p] = (int(row["Pr"]), int(row["Pc"]))
    if not by_p_rep:
        raise SystemExit(f"No rows with N={n} in {csv_path}")
    ps = sorted(by_p_rep)
    by_p = {p: [by_p_rep[p][r] for r in sorted(by_p_rep[p])] for p in ps}
    return ps, by_p, grid


def speedups_by_rep(ps: list[int], by_p: dict[int, list[float]]) -> dict[int, list[float]]:
    if 1 not in by_p:
        raise ValueError("need p=1")
    t1_series = by_p[1]
    out: dict[int, list[float]] = {}
    for p in ps:
        times = by_p[p]
        if len(times) != len(t1_series):
            raise ValueError(f"rep count mismatch at p={p}")
        out[p] = [t1 / tp for t1, tp in zip(t1_series, times)]
    return out


def summarize(ps: list[int], by_p: dict[int, list[float]], t1_mean: float | None = None) -> tuple[float, list[dict]]:
    t1 = t1_mean if t1_mean is not None else statistics.mean(by_p[1])
    rows = []
    for p in ps:
        times = by_p[p]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        sp = t1 / mean_t
        ep = sp / p
        rows.append(
            {
                "p": p,
                "mean_t": mean_t,
                "std_t": std_t,
                "speedup": sp,
                "efficiency": ep,
            }
        )
    return t1, rows


def _box_widths(ps: list[int], frac: float = 0.22) -> list[float]:
    """Multiplicative half-width on log-spaced x."""
    return [max(p * frac, 0.08) for p in ps]


def _draw_boxes(ax: plt.Axes, ps: list[int], series: dict[int, list[float]], fill: str) -> None:
    data = [series[p] for p in ps]
    bp = ax.boxplot(
        data,
        positions=ps,
        widths=_box_widths(ps),
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        zorder=2,
        boxprops=dict(facecolor=fill, edgecolor=BOX_EDGE, linewidth=0.9, alpha=0.92),
        medianprops=dict(color=BOX_EDGE, linewidth=1.2),
        whiskerprops=dict(color=BOX_EDGE, linewidth=0.9),
        capprops=dict(color=BOX_EDGE, linewidth=0.9),
        flierprops=dict(
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=BOX_EDGE,
            alpha=0.85,
        ),
    )
    # matplotlib returns unused; keep reference to silence linters
    _ = bp


def _legend_time() -> list:
    return [
        Patch(facecolor=BOX_FILL_TIME, edgecolor=BOX_EDGE, label="5 runs (box: quartiles)"),
        Line2D([0], [0], color=COLOR_TIME, marker="o", linewidth=1.8, markerfacecolor="white",
               markeredgewidth=1.4, label="Mean $T_p$"),
    ]


def _legend_speedup() -> list:
    return [
        Patch(facecolor=BOX_FILL_SPEEDUP, edgecolor=BOX_EDGE, label="5 runs (box: quartiles)"),
        Line2D([0], [0], color=COLOR_SPEEDUP, marker="s", linewidth=1.8, markerfacecolor="white",
               markeredgewidth=1.4, label="Mean $S_p$ ($T_1^{(r)}/T_p^{(r)}$)"),
        Line2D([0], [0], color=COLOR_IDEAL, linestyle="--", linewidth=1.5, label="Ideal $S_p=p$"),
    ]


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(FIG_BG)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.6, linestyle="-", alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_time(
    ps: list[int],
    by_p: dict[int, list[float]],
    rows: list[dict],
    grid: dict[int, tuple[int, int]],
    out: Path,
    dpi: int,
    n: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1), facecolor=FIG_BG)
    _draw_boxes(ax, ps, by_p, BOX_FILL_TIME)
    means = [r["mean_t"] for r in rows]
    ax.plot(
        ps,
        means,
        "o-",
        color=COLOR_TIME,
        linewidth=1.8,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=4,
    )
    if 2 in ps:
        i = ps.index(2)
        ax.plot(ps[i], means[i], "o", color=COLOR_ANOMALY, markersize=8, zorder=5)
        ax.annotate(
            f"$P_r$={grid[2][0]}, $P_c$={grid[2][1]}",
            (ps[i], means[i]),
            textcoords="offset points",
            xytext=(8, 10),
            fontsize=8,
            color=COLOR_ANOMALY,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("MPI processes $p$")
    ax.set_ylabel("Wall time $T_p$ (s)")
    ax.set_title(f"MPI block matmul wall time ($N={n}$, 5 runs)")
    ax.legend(handles=_legend_time(), loc="upper right", frameon=True, framealpha=0.95, edgecolor=GRID_COLOR)
    _style_axes(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def plot_speedup(
    ps: list[int],
    sp_by_p: dict[int, list[float]],
    rows: list[dict],
    t1: float,
    out: Path,
    dpi: int,
    n: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.1), facecolor=FIG_BG)
    _draw_boxes(ax, ps, sp_by_p, BOX_FILL_SPEEDUP)
    means = [r["speedup"] for r in rows]
    ax.plot(
        ps,
        means,
        "s-",
        color=COLOR_SPEEDUP,
        linewidth=1.8,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=4,
    )
    ideal = [float(p) for p in ps]
    ax.plot(ps, ideal, "--", color=COLOR_IDEAL, linewidth=1.5, zorder=3)

    if 2 in ps:
        i = ps.index(2)
        ax.plot(ps[i], means[i], "s", color=COLOR_ANOMALY, markersize=7, zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("MPI processes $p$")
    ax.set_ylabel("Speedup $S_p$")
    ax.set_title(f"Speedup vs. process count ($N={n}$, mean $T_1$={t1:.1f}\\,s)")
    ax.legend(handles=_legend_speedup(), loc="upper left", frameon=True, framealpha=0.95, edgecolor=GRID_COLOR)
    _style_axes(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def print_table(t1: float, rows: list[dict]) -> None:
    print(f"T1 = {t1:.2f} s (mean over p=1 reps)\n")
    print(f"{'p':>4}  {'T_p':>10}  {'std':>8}  {'S_p':>8}  {'E_p':>8}")
    for r in rows:
        print(
            f"{r['p']:4d}  {r['mean_t']:10.2f}  {r['std_t']:8.2f}  "
            f"{r['speedup']:8.2f}  {r['efficiency']:8.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot matmul benchmark time and speedup")
    ap.add_argument("--csv", type=Path, default=Path("results/matmul_bench.csv"))
    ap.add_argument("--N", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path, default=Path("plots"))
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--time-out", type=Path, default=None, help="default: <out-dir>/matmul_time_vs_procs.png")
    ap.add_argument("--speedup-out", type=Path, default=None, help="default: <out-dir>/matmul_speedup_vs_procs.png")
    args = ap.parse_args()

    ps, by_p, grid = load_bench(args.csv, args.N)
    if 1 not in by_p:
        raise SystemExit("Need p=1 rows to compute T1 and speedup")

    sp_by_p = speedups_by_rep(ps, by_p)
    t1, rows = summarize(ps, by_p)
    print_table(t1, rows)

    time_out = args.time_out or (args.out_dir / "matmul_time_vs_procs.png")
    speedup_out = args.speedup_out or (args.out_dir / "matmul_speedup_vs_procs.png")

    plot_time(ps, by_p, rows, grid, time_out, args.dpi, args.N)
    plot_speedup(ps, sp_by_p, rows, t1, speedup_out, args.dpi, args.N)
    print(f"\nWrote {time_out}")
    print(f"Wrote {speedup_out}")


if __name__ == "__main__":
    main()
