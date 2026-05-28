#!/usr/bin/env bash
# MPI 矩阵乘法基准：每次调用只跑一轮（一个 rep），依次测完全部进程数。
#
# 用法:
#   ./bench_matmul_mpi.sh <rep>
#   ./bench_matmul_mpi.sh <rep> --fresh
#
# 默认只写入 results/matmul_bench.csv（总时间 time_sec）。
# 时间线实验请用 bench_matmul_timeline.sh（不写本 CSV）。
#
# 可选: MATMUL_PROFILE=1  额外在 stdout 输出 PROFILE/RANK（并捕获到 profile CSV）
#
# 相邻两次 mpirun 之间固定等待 10 秒。

set -euo pipefail

INTERVAL_SEC=10
MPIRUN_OPTS="--use-hwthread-cpus"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="$DIR/matmul_mpi"
OUT="$DIR/results/matmul_bench.csv"
OUT_PROFILE="$DIR/results/matmul_bench_profile.csv"
OUT_RANK="$DIR/results/matmul_bench_rank_profile.csv"
MATMUL_N=8000
PROCS="64 32 16 8 4 2 1"

WANT_PROFILE=0
if [[ "${MATMUL_PROFILE:-0}" == "1" ]]; then
  WANT_PROFILE=1
fi

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <rep> [--fresh]" >&2
  exit 1
fi

REP="$1"
FRESH=0
if [[ "${2:-}" == "--fresh" ]]; then
  FRESH=1
fi

if ! [[ "$REP" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误: rep 应为正整数，得到: $REP" >&2
  exit 1
fi

if [[ ! -x "$EXE" ]]; then
  echo "未找到可执行文件，请先编译: cd \"$DIR\" && make" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
if [[ "$FRESH" -eq 1 ]]; then
  rm -f "$OUT"
  [[ "$WANT_PROFILE" -eq 1 ]] && rm -f "$OUT_PROFILE" "$OUT_RANK"
  echo "已删除旧结果: $OUT" >&2
fi
if [[ ! -f "$OUT" ]] || [[ ! -s "$OUT" ]]; then
  echo "N,p,Pr,Pc,rep,time_sec" >"$OUT"
elif [[ "$FRESH" -eq 0 ]]; then
  old_n="$(awk -F, 'NR==2 {print $1; exit}' "$OUT" 2>/dev/null || true)"
  if [[ -n "$old_n" && "$old_n" != "$MATMUL_N" ]]; then
    echo "错误: 已有 CSV 中 N=$old_n，与本次 N=$MATMUL_N 不一致。" >&2
    exit 1
  fi
fi
if [[ "$WANT_PROFILE" -eq 1 ]] && [[ ! -f "$OUT_PROFILE" ]]; then
  echo "N,p,Pr,Pc,rep,time_sec,scatter_a_sec,scatter_b_sec,k_comm_sec,k_gemm_sec,gather_sec,other_sec" >"$OUT_PROFILE"
  echo "N,p,Pr,Pc,rep,rank,scatter_a_sec,scatter_b_sec,k_comm_sec,k_gemm_sec,gather_sec" >"$OUT_RANK"
fi

wall_start=$(date +%s)
echo "开始第 ${REP} 轮, N=$MATMUL_N, PROCS=[$PROCS]" >&2

first_p=1
for p in $PROCS; do
  if [[ "$first_p" -eq 0 ]]; then
    sleep "$INTERVAL_SEC"
  fi
  first_p=0
  p_start=$(date +%s)

  if [[ "$WANT_PROFILE" -eq 1 ]]; then
    mpi_out="$(mpirun $MPIRUN_OPTS -np "$p" "$EXE" "$MATMUL_N" 2>/dev/null)" || true
    line="$(printf '%s\n' "$mpi_out" | head -1)"
  else
    line="$(mpirun $MPIRUN_OPTS -np "$p" "$EXE" "$MATMUL_N" 2>/dev/null | head -1)" || true
  fi

  if [[ -z "$line" ]]; then
    echo "错误: 第 ${REP} 轮 p=$p 无输出，mpirun 报错如下:" >&2
    mpirun $MPIRUN_OPTS -np "$p" "$EXE" "$MATMUL_N" >&2 || true
    exit 1
  fi
  read -r n_out p_out pr_out pc_out t_out <<<"$line"
  if [[ "$n_out" != "$MATMUL_N" ]]; then
    echo "错误: 第 ${REP} 轮 p=$p 输出 N=$n_out，期望 $MATMUL_N" >&2
    exit 1
  fi
  echo "$n_out,$p_out,$pr_out,$pc_out,$REP,$t_out" >>"$OUT"

  if [[ "$WANT_PROFILE" -eq 1 ]]; then
    profile_line="$(printf '%s\n' "$mpi_out" | grep '^PROFILE ' | head -1 || true)"
    if [[ -n "$profile_line" ]]; then
      read -r _prof sa sb kc kg tg oth <<<"$profile_line"
      echo "$n_out,$p_out,$pr_out,$pc_out,$REP,$t_out,$sa,$sb,$kc,$kg,$tg,$oth" >>"$OUT_PROFILE"
      while read -r rk sa sb kc kg tg; do
        [[ -z "$rk" ]] && continue
        echo "$n_out,$p_out,$pr_out,$pc_out,$REP,$rk,$sa,$sb,$kc,$kg,$tg" >>"$OUT_RANK"
      done < <(printf '%s\n' "$mpi_out" | grep '^RANK ')
    fi
  fi

  p_elapsed=$(($(date +%s) - p_start))
  echo "  第 ${REP} 轮 p=$p  time_sec=$t_out  墙钟=${p_elapsed}s" >&2
done

wall_elapsed=$(($(date +%s) - wall_start))
echo "第 ${REP} 轮完成: 墙钟 ${wall_elapsed}s (约 $(awk "BEGIN{printf \"%.1f\", $wall_elapsed/60}") min)" >&2
echo "结果已追加: $OUT" >&2
