# Assignment 3：MPI 二维块划分矩阵乘法

使用 **MPI** 实现 \(N\times N\) 双精度方阵乘法的二维块划分：\(C_{i,j}=\sum_k A_{i,k}B_{k,j}\)，进程网格 \(P_r\times P_c=p\)。计时段含子块分发、按 \(k\) 轮阻塞式 `MPI_Send`/`MPI_Recv` 与本地乘加、以及向 rank 0 汇聚 \(C\)；矩阵初始化在 `MPI_Wtime` 之前完成。

报告与说明：`并行程序设计A_h3.pdf`、`Exercise-3.pdf`。

## 目录说明

| 路径 | 说明 |
|------|------|
| `并行程序设计A_h3.pdf` | 课程作业报告 |
| `Exercise-3.pdf` | 实验说明 |
| `matmul_mpi.c` | 主程序 |
| `Makefile` | `mpicc -O3` 编译，本地生成 `matmul_mpi` |
| `bench_matmul_mpi.sh` | 五轮基准，写入 `results/matmul_bench.csv` |
| `bench_matmul_timeline.sh` | 时间线实验，`MATMUL_EVENTS=1` |
| `plot_matmul_bench.py` | 绘制耗时 / 加速比 |
| `plot_timeline.py` | 绘制各 \(p\) 的 rank 时间线 |
| `analyze_timeline.py` | 可选：终端打印单组事件时间线 |
| `results/matmul_bench.csv` | 基准原始数据 |
| `results/matmul_bench_timeline.csv` | 开启事件记录时的墙钟 |
| `results/events/` | 每 rank 事件 CSV：`N{N}_p{p}_rep{r}_rank{rrr}.csv` |
| `plots/` | 脚本生成的 PNG 图表 |

提交前在本机 `make`，勿将可执行文件、LaTeX 报告等与源码一并提交。

## 依赖

- Open MPI：`mpicc`、`mpirun`
- Python 3 + **matplotlib**（绘图）
- 实验环境建议：`mpirun --use-hwthread-cpus`（与课程容器逻辑 CPU 数一致）

## 编译

```bash
cd h3
make
```

## 运行

### 五轮基准

```bash
./bench_matmul_mpi.sh 1 --fresh   # 第 1 轮，清空同 rep 旧数据
./bench_matmul_mpi.sh 2
./bench_matmul_mpi.sh 3
./bench_matmul_mpi.sh 4
./bench_matmul_mpi.sh 5
```

每轮内依次测试 \(p=64,32,16,8,4,2,1\)，相邻两次 `mpirun` 间隔 10s。默认不开启事件记录。

### 单次运行

```bash
mpirun --use-hwthread-cpus -np <p> ./matmul_mpi 8000
```

rank 0 输出一行：`N p Pr Pc time_sec`。

### 时间线实验

与基准分离，不写 `matmul_bench.csv`：

```bash
./bench_matmul_timeline.sh --fresh          # 全部 p
./bench_matmul_timeline.sh --p 8 --fresh  # 仅 p=8
```

### 绘图

```bash
python3 plot_matmul_bench.py
python3 plot_timeline.py --batch --events-dir results/events --N 8000 --rep 1
```

时间线图默认最大高度约 8.5 英寸，可用 `--max-height` 调整。

## 环境变量

| 变量 | 含义 |
|------|------|
| `MATMUL_EVENTS=1` | 计时段内事件写入 `results/events/` |
| `MATMUL_REP=<n>` | 事件文件名中的 rep 编号，默认 1 |
| `MATMUL_EVENTS_DIR=...` | 事件目录，默认 `results/events` |
| `MATMUL_PROFILE=1` | stdout 输出各阶段耗时汇总 |
