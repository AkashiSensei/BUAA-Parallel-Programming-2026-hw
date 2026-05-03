#!/usr/bin/env bash
set -euo pipefail

# 在本文件顶部直接改这些变量即可；也支持在运行前用环境变量覆盖同名变量
N="${N:-200000000}"
THREADS="${THREADS:-1 2 3 4 6 8}"
RUNS="${RUNS:-5}"
SEED="${SEED:-1}"
THRESHOLD="${THRESHOLD:-16384}"

PQS_BIN="${PQS_BIN:-./pqs}"
OUT_CSV="${OUT_CSV:-./results/pqs_bench.csv}"
AUTO_BUILD="${AUTO_BUILD:-1}" # 1 表示缺少 pqs 时自动 make

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" == "0" ]]; then
  echo "无效的 N: $N" >&2
  exit 2
fi
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" == "0" ]]; then
  echo "无效的 RUNS: $RUNS" >&2
  exit 2
fi

if [[ ! -f "$PQS_BIN" ]]; then
  if [[ "$AUTO_BUILD" == "1" ]] && command -v make >/dev/null 2>&1; then
    make
  else
    echo "未找到 $PQS_BIN，且 AUTO_BUILD!=1 或没有 make" >&2
    exit 1
  fi
fi

if [[ ! -f "$PQS_BIN" ]]; then
  echo "仍未找到 $PQS_BIN" >&2
  exit 1
fi

if [[ ! -x "$PQS_BIN" ]]; then
  chmod +x "$PQS_BIN" 2>/dev/null || true
fi
if [[ ! -x "$PQS_BIN" ]]; then
  echo "$PQS_BIN 不可执行（请 chmod +x 或重新 make）" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

echo "n,threads,run,time_sec,seed,threshold" >"$tmp"

parse_time_sec() {
  local line="$1"
  if [[ "$line" =~ time_sec=([0-9]+\.[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

echo "bench: n=$N runs=$RUNS seed=$SEED threshold=$THRESHOLD pqs=$PQS_BIN"
echo "threads: $THREADS"
echo "csv: $OUT_CSV"
echo

for t in $THREADS; do
  if ! [[ "$t" =~ ^[0-9]+$ ]] || [[ "$t" == "0" ]]; then
    echo "无效的线程数: $t" >&2
    exit 2
  fi

  sum=0
  printf '%s\n' "--- threads=$t ---"
  for ((i = 1; i <= RUNS; i++)); do
    line="$("$PQS_BIN" -n "$N" -t "$t" --seed "$SEED" --threshold "$THRESHOLD")"
    ts="$(parse_time_sec "$line")" || {
      echo "无法解析输出: $line" >&2
      exit 1
    }
    printf 'run %d/%d: %s\n' "$i" "$RUNS" "$line"
    echo "$N,$t,$i,$ts,$SEED,$THRESHOLD" >>"$tmp"
    sum="$(awk -v a="$sum" -v b="$ts" 'BEGIN{printf "%.12f", a+b}')"
  done

  avg="$(awk -v s="$sum" -v r="$RUNS" 'BEGIN{printf "%.6f", s/r}')"
  printf 'avg threads=%s: %s sec\n\n' "$t" "$avg"
done

cp "$tmp" "$OUT_CSV"
echo "已写入: $OUT_CSV"
