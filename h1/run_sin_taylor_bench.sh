#!/usr/bin/env bash
# sin_taylor_omp：x、项数见下方常量；线程 0 / 2 / 4 / 8；结果写入 results/sin_taylor_bench.txt
# 可选参数 1：输出文件路径（默认 results/sin_taylor_bench.txt）
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$DIR/sin_taylor_omp"
RES_DIR="$DIR/results"

X=1.0
N_TERMS=300000000000
OUT="${1:-$RES_DIR/sin_taylor_bench.txt}"

if [[ ! -x "$EXE" ]]; then
  echo "未找到可执行文件，请先编译: cd \"$DIR\" && make sin_taylor_omp" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
: >"$OUT"
for t in 0 2 4 8; do
  "$EXE" "$X" "$N_TERMS" "$t" >>"$OUT"
done
echo "结果已写入: ${OUT} (x=${X}, n_terms=${N_TERMS})" >&2
