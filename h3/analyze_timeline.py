#!/usr/bin/env python3
"""从 matmul_bench_events.csv 重建各 rank 时间线（含事件间隙标为 idle）。"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_events(path: Path, rep: int, p: int):
    rows = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["rep"]) != rep or int(row["p"]) != p:
                continue
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="打印 MPI 矩阵乘事件时间线")
    ap.add_argument(
        "csv",
        nargs="?",
        default="results/matmul_bench_events.csv",
        help="事件 CSV 路径",
    )
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--p", type=int, required=True, help="进程数")
    ap.add_argument("--rank", type=int, default=None, help="只显示该 rank")
    ap.add_argument("--gap-ms", type=float, default=0.01, help="间隙超过此值(秒)标 idle")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"未找到: {path}")

    rows = load_events(path, args.rep, args.p)
    if not rows:
        raise SystemExit(f"无 rep={args.rep} p={args.p} 的数据")

    by_rank = defaultdict(list)
    for row in rows:
        by_rank[int(row["rank"])].append(row)
    for rk in by_rank:
        by_rank[rk].sort(key=lambda x: int(x["seq"]))

    ranks = [args.rank] if args.rank is not None else sorted(by_rank)
    for rk in ranks:
        evs = by_rank.get(rk, [])
        if not evs:
            continue
        print(f"\n=== rank {rk} ({len(evs)} events) ===")
        prev_end = 0.0
        for e in evs:
            t0 = float(e["t_start"])
            t1 = float(e["t_end"])
            if t0 - prev_end > args.gap_ms:
                print(
                    f"  [{prev_end:10.6f},{t0:10.6f})  idle/other  "
                    f"gap={t0 - prev_end:.6f}s"
                )
            peer = e["peer"]
            k = e["k"]
            print(
                f"  [{t0:10.6f},{t1:10.6f})  {e['phase']:10} {e['kind']:8}  "
                f"k={k} peer={peer} tag={e['tag']}  dur={e['dur_sec']}s"
            )
            prev_end = t1
        total = float(evs[-1]["t_end"])
        print(f"  rank 末时刻 ≈ {total:.6f}s")


if __name__ == "__main__":
    main()
