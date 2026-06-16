#!/usr/bin/env bash
set -euo pipefail

UTIL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEMM_DIR="${GEMM_DIR:-.}"
DATA_DIR="${DATA_DIR:-$GEMM_DIR/../data}"
RESULTS_DIR="${RESULTS_DIR:-$GEMM_DIR/results}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
BLOCK="${BLOCK:-16}"
SIZES="${SIZES:-512 1024 2048 4096 8192}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
IMPL_NAME="${IMPL_NAME:-$(basename "$GEMM_DIR")}"

DETAIL_CSV="${DETAIL_CSV:-$RESULTS_DIR/bench_detail.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULTS_DIR/bench_summary.csv}"

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$GEMM_DIR"
make -s bench
mkdir -p "$(dirname "$DETAIL_CSV")"

{
  echo "impl,n,block,warmup,iter,time_ms"
} > "$DETAIL_CSV"
{
  echo "impl,n,block,warmup,iters,mean_ms,std_ms,var_ms,min_ms,max_ms,mean_gflops"
} > "$SUMMARY_CSV"

echo "# $IMPL_NAME bench warmup=$WARMUP iters=$ITERS block=${BLOCK}x${BLOCK} gpu=$GPU"
echo "# detail_csv=$DETAIL_CSV"
echo "# summary_csv=$SUMMARY_CSV"
for n in $SIZES; do
  ./bench -n "$n" --data-dir "$DATA_DIR" --warmup "$WARMUP" --iters "$ITERS" --block "$BLOCK" \
    --impl "$IMPL_NAME" --csv-detail "$DETAIL_CSV" --csv-summary "$SUMMARY_CSV"
done
