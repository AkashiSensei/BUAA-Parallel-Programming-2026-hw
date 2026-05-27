#!/usr/bin/env python3
"""Plot per-rank MPI event timelines (PNG, English). Times aligned to rank 0 origin."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Custom semantic colors (user preference)
KIND_COLOR = {
    "compute": "#4e79a7",  # blue
    "send": "#f28e2b",     # orange
    "recv": "#bab0ac",     # grey (represents waiting)
    "pack": "#e15759",     # red
    "alloc": "#b07aa1",    # purple
}

FIG_BG = "#FFFFFF"
TRACK_BG = "#F5F7FA"
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

KIND_LABEL = {
    "compute": "compute",
    "send": "send",
    "recv": "recv (incl. wait)",
    "pack": "pack (local copy)",
    "alloc": "alloc (memory)",
}


def load_events(path: Path, rep: int | None, p: int) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["p"]) != p:
                continue
            if rep is not None and int(row["rep"]) != rep:
                continue
            rows.append(row)
    return rows


def event_segments(evs: list[dict]) -> list[dict]:
    """Only recorded events; gaps (incl. before first event) stay blank."""
    segs = []
    for e in sorted(evs, key=lambda x: int(x["seq"])):
        t0 = float(e["t_start"])
        t1 = float(e["t_end"])
        if t1 <= t0:
            continue
        if t1 <= 0:
            continue
        if t0 < 0:
            t0 = 0.0
        segs.append({"t0": t0, "t1": t1, "kind": e["kind"]})
    return segs


def render_png(
    meta: dict,
    by_rank: dict[int, list],
    out_path: Path,
    dpi: int = 150,
) -> None:
    rank_segs = {rk: event_segments(evs) for rk, evs in sorted(by_rank.items())}
    ranks = sorted(rank_segs.keys())
    t_max = 0.0
    for segs in rank_segs.values():
        for s in segs:
            t_max = max(t_max, s["t1"])
    if t_max <= 0:
        t_max = 1.0

    row_h = 0.85
    fig_h = max(4.0, 1.2 + row_h * len(ranks))
    fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=FIG_BG)
    ax.set_facecolor("white")

    for i, rk in enumerate(ranks):
        y = len(ranks) - 1 - i
        ax.barh(
            y,
            t_max * 1.02,
            left=0,
            height=0.78,
            color=TRACK_BG,
            edgecolor="none",
            align="center",
            zorder=0,
        )
        for s in rank_segs[rk]:
            w = s["t1"] - s["t0"]
            ax.barh(
                y,
                w,
                left=s["t0"],
                height=0.62,
                color=KIND_COLOR.get(s["kind"], "#717581"),
                edgecolor="white",
                linewidth=0.35,
                align="center",
                zorder=2,
            )

    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"rank {rk}" for rk in ranks])
    ax.set_xlabel("Time since rank 0 timed-section start (s, MPI_Wtime)")
    ax.set_xlim(0, t_max * 1.02)
    ax.set_title(
        f"MPI block matmul timeline  N={meta['N']}  p={meta['p']}  "
        f"grid={meta['Pr']}x{meta['Pc']}  rep={meta.get('rep', '?')}",
        fontsize=12,
        fontweight="medium",
        pad=10,
    )
    ax.grid(axis="x", linestyle="-", color=GRID_COLOR, alpha=0.65, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)

    handles = [
        Patch(facecolor=KIND_COLOR[k], edgecolor="white", linewidth=0.5, label=KIND_LABEL[k])
        for k in ("compute", "send", "recv", "pack", "alloc")
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor=GRID_COLOR,
        framealpha=0.95,
    )

    fig.text(
        0.01,
        0.01,
        "Blank = no activity; recv includes blocking wait.",
        fontsize=7.5,
        color="#6B7B8C",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MPI event timeline (PNG)")
    ap.add_argument("csv", nargs="?", default="results/matmul_bench_events.csv")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("-o", "--output", default="plots/timeline_demo.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    path = Path(args.csv)
    rows = load_events(path, args.rep, args.p)
    if not rows:
        raise SystemExit(f"No data: {path} rep={args.rep} p={args.p}")

    by_rank: dict[int, list] = defaultdict(list)
    for row in rows:
        by_rank[int(row["rank"])].append(row)

    meta = {
        "N": rows[0]["N"],
        "p": rows[0]["p"],
        "Pr": rows[0]["Pr"],
        "Pc": rows[0]["Pc"],
        "rep": rows[0]["rep"],
    }
    render_png(meta, by_rank, Path(args.output), dpi=args.dpi)


if __name__ == "__main__":
    main()
