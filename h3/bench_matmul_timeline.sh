#!/usr/bin/env bash
# 时间线实验：对每个进程数 p 单独跑一次，各 rank 写入 results/events/*.csv
# 不修改 results/matmul_bench.csv；总时间写入 matmul_bench_timeline.csv（含 profiling 开销）。
#
# 用法:
#   ./bench_matmul_timeline.sh              # 全部 p，rep=1
#   ./bench_matmul_timeline.sh --fresh      # 删除本次 N/rep 的旧事件 CSV 后再跑
#   ./bench_matmul_timeline.sh --p 8        # 只跑 p=8
#   ./bench_matmul_timeline.sh --rep 2      # 文件名中用 rep=2
#
# 环境（一般无需改）:
#   MATMUL_EVENTS=1（脚本内已 export）
#   MATMUL_REP、MATMUL_EVENTS_DIR

set -euo pipefail

INTERVAL_SEC=10
MPIRUN_OPTS="--use-hwthread-cpus"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$DIR/matmul_mpi"
EVENTS_DIR="${MATMUL_EVENTS_DIR:-$DIR/results/events}"
OUT_TIMES="$DIR/results/matmul_bench_timeline.csv"
MATMUL_N=8000
PROCS="64 32 16 8 4 2 1"
REP=1
FRESH=0
ONLY_P=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH=1
      ;;
    --rep)
      REP="${2:?缺少 --rep 参数}"
      shift
      ;;
    --p)
      ONLY_P="${2:?缺少 --p 参数}"
      shift
      ;;
    -h | --help)
      sed -n '2,14p' "$0" >&2
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -n "$ONLY_P" ]]; then
  PROCS="$ONLY_P"
fi

if ! [[ "$REP" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误: rep 应为正整数" >&2
  exit 1
fi

if [[ ! -x "$EXE" ]]; then
  echo "未找到 $EXE，请先 make" >&2
  exit 1
fi

mkdir -p "$EVENTS_DIR" "$(dirname "$OUT_TIMES")"

if [[ "$FRESH" -eq 1 ]]; then
  pattern="$EVENTS_DIR/N${MATMUL_N}_p*_rep${REP}_rank*.csv"
  # shellcheck disable=SC2086
  rm -f $pattern 2>/dev/null || true
  rm -f "$OUT_TIMES"
  echo "已删除事件 CSV 与 $OUT_TIMES" >&2
fi
if [[ ! -f "$OUT_TIMES" ]] || [[ ! -s "$OUT_TIMES" ]]; then
  echo "N,p,Pr,Pc,rep,time_sec" >"$OUT_TIMES"
fi

export MATMUL_EVENTS=1
export MATMUL_REP="$REP"
export MATMUL_EVENTS_DIR="$EVENTS_DIR"

echo "时间线实验 N=$MATMUL_N rep=$REP PROCS=[$PROCS]" >&2
echo "事件目录: $EVENTS_DIR" >&2

first_p=1
for p in $PROCS; do
  if [[ "$first_p" -eq 0 ]]; then
    sleep "$INTERVAL_SEC"
  fi
  first_p=0
  p_start=$(date +%s)

  mpi_out="$(mpirun $MPIRUN_OPTS -np "$p" "$EXE" "$MATMUL_N" 2>/dev/null)" || true
  line="$(printf '%s\n' "$mpi_out" | head -1)"
  events_line="$(printf '%s\n' "$mpi_out" | grep '^EVENTS_DIR ' | head -1 || true)"

  if [[ -z "$line" ]]; then
    echo "错误: p=$p 无输出，mpirun 报错:" >&2
    mpirun $MPIRUN_OPTS -np "$p" "$EXE" "$MATMUL_N" >&2 || true
    exit 1
  fi

  read -r n_out p_out pr_out pc_out t_out <<<"$line"
  if [[ "$n_out" != "$MATMUL_N" ]]; then
    echo "错误: p=$p 输出 N=$n_out，期望 $MATMUL_N" >&2
    exit 1
  fi
  echo "$n_out,$p_out,$pr_out,$pc_out,$REP,$t_out" >>"$OUT_TIMES"

  n_files="$(find "$EVENTS_DIR" -maxdepth 1 -name "N${MATMUL_N}_p${p}_rep${REP}_rank*.csv" 2>/dev/null | wc -l)"
  p_elapsed=$(($(date +%s) - p_start))

  echo "  p=$p  time_sec=$t_out  event_files=$n_files  ${events_line:-}" >&2
  echo "  墙钟=${p_elapsed}s" >&2
done

echo "总时间（含 MATMUL_EVENTS profiling）已写入: $OUT_TIMES" >&2
echo "可与基准对比: results/matmul_bench.csv（同列 time_sec）" >&2
echo "画图示例:" >&2
echo "  python3 plot_timeline.py --batch --events-dir $EVENTS_DIR --N $MATMUL_N --rep $REP" >&2
echo "  # or single: plot_timeline.py --events-dir $EVENTS_DIR --N $MATMUL_N --p 8 --rep $REP -o plots/timeline_p8.png" >&2
