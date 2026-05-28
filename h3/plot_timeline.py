#!/usr/bin/env python3
"""Plot per-rank MPI event timelines (PNG, English). Times aligned to rank 0 origin."""

from __future__ import annotations

import argparse
import csv
import math
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

# Bar heights in y-axis data units (one rank = 1.0 spacing)
TRACK_H = 0.78
BAR_H = 0.62

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

DEFAULT_P_LIST = [1, 2, 4, 8, 16, 32, 64]


def load_events_csv(path: Path, rep: int | None, p: int) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["p"]) != p:
                continue
            if rep is not None and int(row["rep"]) != rep:
                continue
            rows.append(row)
    return rows


def load_events_dir(events_dir: Path, n: int, rep: int, p: int) -> list[dict]:
    pattern = f"N{n}_p{p}_rep{rep}_rank*.csv"
    rows: list[dict] = []
    for path in sorted(events_dir.glob(pattern)):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def discover_p_values(events_dir: Path, n: int, rep: int) -> list[int]:
    ps: set[int] = set()
    for path in events_dir.glob(f"N{n}_p*_rep{rep}_rank*.csv"):
        # N8000_p64_rep1_rank000.csv
        parts = path.stem.split("_")
        for part in parts:
            if part.startswith("p") and part[1:].isdigit():
                ps.add(int(part[1:]))
                break
    return sorted(ps)


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


def figure_layout(
    n_ranks: int,
    *,
    max_fig_height: float = 8.5,
    min_fig_height: float = 3.2,
    max_row_inches: float = 0.72,
    min_row_inches: float = 0.09,
    margin_inches: float = 1.2,
) -> tuple[float, float]:
    """Return (fig_height_inches, row_inches) with capped total height."""
    if n_ranks <= 0:
        return min_fig_height, max_row_inches
    # Shrink per-rank physical height as p grows so fig_h stays bounded.
    row_inches = (max_fig_height - margin_inches) / n_ranks
    row_inches = min(max_row_inches, max(min_row_inches, row_inches))
    fig_h = min(max_fig_height, max(min_fig_height, margin_inches + row_inches * n_ranks))
    return fig_h, row_inches


def ytick_labels(ranks: list[int], max_labeled: int = 20) -> list[str]:
    n = len(ranks)
    if n <= max_labeled:
        return [f"rank {rk}" for rk in ranks]
    step = max(1, math.ceil(n / max_labeled))
    labels: list[str] = []
    for i, rk in enumerate(ranks):
        if i == 0 or i == n - 1 or i % step == 0:
            labels.append(f"rank {rk}")
        else:
            labels.append("")
    return labels


def render_png(
    meta: dict,
    by_rank: dict[int, list],
    out_path: Path,
    dpi: int = 150,
    *,
    max_fig_height: float = 8.5,
    min_fig_height: float = 3.2,
    max_row_inches: float = 0.72,
    min_row_inches: float = 0.09,
    margin_inches: float = 1.2,
    max_ytick_labels: int = 20,
) -> None:
    rank_segs = {rk: event_segments(evs) for rk, evs in sorted(by_rank.items())}
    ranks = sorted(rank_segs.keys())
    t_max = 0.0
    for segs in rank_segs.values():
        for s in segs:
            t_max = max(t_max, s["t1"])
    if t_max <= 0:
        t_max = 1.0

    n = len(ranks)
    fig_h, row_inches = figure_layout(
        n,
        max_fig_height=max_fig_height,
        min_fig_height=min_fig_height,
        max_row_inches=max_row_inches,
        min_row_inches=min_row_inches,
        margin_inches=margin_inches,
    )
    y_font = max(5.5, min(9.0, 7.5 * (0.22 / max(row_inches, 0.08))))

    fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=FIG_BG)
    ax.set_facecolor("white")

    for i, rk in enumerate(ranks):
        y = len(ranks) - 1 - i
        ax.barh(
            y,
            t_max * 1.02,
            left=0,
            height=TRACK_H,
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
                height=BAR_H,
                color=KIND_COLOR.get(s["kind"], "#717581"),
                edgecolor="white",
                linewidth=0.35,
                align="center",
                zorder=2,
            )

    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(ytick_labels(ranks, max_labeled=max_ytick_labels), fontsize=y_font)
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

    note = "Blank = no activity; recv includes blocking wait."
    if n > 20:
        note += f" ({n} ranks; y-axis labels subsampled)."
    fig.text(0.01, 0.01, note, fontsize=7.0, color="#6B7B8C")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}  ({n} ranks, fig_h={fig_h:.1f} in, row={row_inches:.2f} in)")


def plot_one(
    events_dir: Path,
    n: int,
    rep: int,
    p: int,
    out_path: Path,
    dpi: int,
    **render_kw,
) -> bool:
    rows = load_events_dir(events_dir, n, rep, p)
    if not rows:
        print(f"Skip p={p}: no event files under {events_dir}")
        return False
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
    render_png(meta, by_rank, out_path, dpi=dpi, **render_kw)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MPI event timeline (PNG)")
    ap.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Legacy merged events CSV (optional)",
    )
    ap.add_argument(
        "--events-dir",
        default="results/events",
        help="Per-rank event CSV directory (default)",
    )
    ap.add_argument("--N", type=int, default=8000, help="Matrix size (with --events-dir)")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--p", type=int, default=None, help="MPI process count (required unless --batch)")
    ap.add_argument("-o", "--output", default=None, help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--max-height",
        type=float,
        default=8.5,
        help="Cap figure height in inches (default 8.5, balance report page and readability)",
    )
    ap.add_argument(
        "--max-row-inches",
        type=float,
        default=0.72,
        help="Max physical height per rank for small p (default 0.72)",
    )
    ap.add_argument(
        "--min-row-inches",
        type=float,
        default=0.09,
        help="Min physical height per rank for large p (default 0.09)",
    )
    ap.add_argument(
        "--batch",
        action="store_true",
        help="Plot all p with event files (or --p-list)",
    )
    ap.add_argument(
        "--p-list",
        default="1,2,4,8,16,32,64",
        help="Comma-separated p values for --batch",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for --batch (timeline_p<p>.png)",
    )
    args = ap.parse_args()

    render_kw = {
        "max_fig_height": args.max_height,
        "max_row_inches": args.max_row_inches,
        "min_row_inches": args.min_row_inches,
    }
    events_dir = Path(args.events_dir)

    if args.batch:
        if args.p_list.strip().lower() == "auto":
            ps = discover_p_values(events_dir, args.N, args.rep)
        else:
            ps = [int(x.strip()) for x in args.p_list.split(",") if x.strip()]
        if not ps:
            raise SystemExit("No p values to plot")
        ok = 0
        for p in ps:
            out = args.out_dir / f"timeline_p{p}.png"
            if plot_one(events_dir, args.N, args.rep, p, out, args.dpi, **render_kw):
                ok += 1
        print(f"Done: {ok}/{len(ps)} timelines in {args.out_dir}")
        return

    if args.p is None:
        raise SystemExit("Specify --p or use --batch")

    out_path = Path(args.output) if args.output else Path(f"plots/timeline_p{args.p}.png")

    if args.csv is not None:
        rows = load_events_csv(Path(args.csv), args.rep, args.p)
        src = str(args.csv)
        if not rows:
            raise SystemExit(f"No data: {src} rep={args.rep} p={args.p}")
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
        render_png(meta, by_rank, out_path, dpi=args.dpi, **render_kw)
    else:
        if not plot_one(events_dir, args.N, args.rep, args.p, out_path, args.dpi, **render_kw):
            raise SystemExit(
                f"No data: {events_dir}/N{args.N}_p{args.p}_rep{args.rep}_rank*.csv"
            )


if __name__ == "__main__":
    main()
