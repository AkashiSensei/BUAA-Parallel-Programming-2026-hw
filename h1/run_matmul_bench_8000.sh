#!/usr/bin/env bash
# 矩阵乘法 matmul_omp：N=8000，线程 0 / 2 / 4 / 8；结果写入 OUT（默认同目录 matmul_bench_8000.txt）
# 可执行文件若已按绝对路径链接 libomp（见 otool -L matmul_omp），无需 DYLD_LIBRARY_PATH。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$DIR/matmul_omp"
OUT="${1:-$DIR/results/matmul_bench_8000.txt}"

if [[ ! -x "$EXE" ]]; then
  echo "未找到可执行文件，请先编译: cd \"$DIR\" && make" >&2
  exit 1
fi

N=8000
: >"$OUT"
for t in 0 2 4 8; do
  "$EXE" "$N" "$t" >>"$OUT"
done
echo "结果已写入: $OUT" >&2
