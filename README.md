# BUAA-Parallel-Programming-2026-hw

2026 年北航研究生《并行程序设计 A》（刘轶老师-D061251001）课程作业。

本仓库按作业分目录存放源码、实验脚本、原始测数与绘图结果；各次作业说明以对应 PDF 为准。

---

## Assignment 1（h1）：OpenMP 入门 — 并行矩阵乘法与泰勒级数求和

围绕 **OpenMP** 的并行循环与线程数控制，完成两个数值实验：**N×N 方阵乘法**与 **固定 x 下 sin(x) 泰勒展开项的并行求和**（含串行 baseline：`threads=0`）。配套脚本批量跑不同线程数、记录一行式输出，并用 Python 生成耗时与加速比曲线。

```
h1/
├── 并行程序设计A_h1.pdf             # 课程作业报告
├── Exercise-1.pdf                 # 实验说明 PDF
├── matmul_omp.c                   # 并行方阵乘法（OpenMP parallel for）
├── sin_taylor_omp.c               # sin 泰勒展开按项并行求和（区间划分 + 递推）
├── Makefile                       # clang + libomp 编译两个可执行文件
├── run_matmul_bench_8000.sh       # 矩阵乘法基准：N=8000，线程 0/2/4/8
├── run_sin_taylor_bench.sh        # 泰勒求和基准脚本
├── plot_experiment_results.py     # 读取 results/ 下文本并生成 plots/ 图表
├── plots/
│   ├── matmul_execution_time.png  # 矩阵乘法执行时间
│   ├── matmul_speedup.png         # 矩阵乘法加速比
│   ├── sin_taylor_execution_time.png
│   └── sin_taylor_speedup.png
└── results/
    ├── matmul_bench_8000_*.txt    # 矩阵乘法各次运行原始输出
    └── sin_taylor_bench*.txt      # 泰勒实验原始输出
```

---

## Assignment 2（h2）：pthread 并行快速排序与可扩展性测试

实现基于 **pthread** 的 **并行快速排序**：Hoare 风格划分、任务栈调度、工作线程从栈中取子区间；子问题小于阈值时回落到 `qsort` 串行排序。`bench_pqs.sh` 对不同线程数重复运行并写入 CSV，`plot_pqs_bench.py` 绘制耗时与加速比。

```
h2/
├── 并行程序设计A_h2.pdf             # 课程作业报告
├── Exercise-2.pdf                 # 实验说明 PDF
├── pqs.c                          # 并行快排主程序（互斥锁 + 条件变量管理任务栈）
├── Makefile                       # 编译生成 pqs（-pthread）
├── bench_pqs.sh                   # 批量基准：可调 N、线程列表、重复次数、阈值等
├── plot_pqs_bench.py              # 读取 results/pqs_bench.csv 并生成图表
├── plots/
│   ├── pqs_time_vs_threads.png    # 排序耗时随线程数变化
│   └── pqs_speedup_vs_threads.png # 加速比随线程数变化
└── results/
    └── pqs_bench.csv              # 基准原始数据（CSV）
```

---

## Assignment 3（h3）：MPI 二维块划分矩阵乘法

使用 **MPI** 实现 \(N\times N\) 双精度方阵乘法的二维块划分，按进程数考察强扩展；配套脚本完成五轮基准、可选事件时间线记录，并用 Python 绘制耗时、加速比与各 rank Gantt 图。编译与运行见 [h3/README.md](h3/README.md)。

```
h3/
├── 并行程序设计A_h3.pdf             # 课程作业报告
├── Exercise-3.pdf                 # 实验说明 PDF
├── matmul_mpi.c                   # MPI 块矩阵乘法主程序
├── Makefile                       # mpicc -O3 编译
├── bench_matmul_mpi.sh            # 五轮基准脚本
├── bench_matmul_timeline.sh       # 时间线实验脚本
├── plot_matmul_bench.py           # 绘制耗时 / 加速比
├── plot_timeline.py               # 绘制各 p 的 rank 时间线
├── analyze_timeline.py            # 可选：终端打印事件时间线
├── README.md                      # 编译、运行与环境变量说明
├── plots/
│   ├── matmul_time_vs_procs.png
│   ├── matmul_speedup_vs_procs.png
│   └── timeline_p{1,2,4,8,16,32,64}.png
└── results/
    ├── matmul_bench.csv
    ├── matmul_bench_timeline.csv
    └── events/
```

---

## Final（final）：CUDA GEMM 优化与性能分析

围绕 **CUDA 双精度矩阵乘法（GEMM）** 完成 GPU kernel 实现、正确性校验、benchmark 与 Nsight profiling 分析。实验从一线程计算一个输出元素的朴素实现出发，对比关闭优化（`-O0`）与开启优化（`-O3`）的指令生成差异；随后实现基于 **shared memory tiled GEMM** 的版本，并加入 **cuBLAS** 作为标准库性能参照。报告中结合 Nsight Systems / Nsight Compute、Roofline、存储与缓存、调度发射、occupancy、stall reason、SASS 指令流程等信息进行横向分析。

```text
final/
├── 并行程序设计A_Final.pdf                 # 渲染后的最终实验报告
├── Final-2026.pdf                         # 课程 final 作业说明 PDF
├── .gitignore                             # 忽略中间可执行文件与大规模输入矩阵数据
├── data/                                  # 输入矩阵与 cuBLAS 参考结果生成工具
│   ├── Makefile
│   ├── gen_data.c                         # 生成随机 A/B 输入数据
│   └── gen_cref.cu                        # 调用 cuBLAS 生成 C_ref
├── util/                                  # 公共 benchmark / verify / Makefile 规则
│   ├── bench.cu                           # cudaEvent 计时 benchmark
│   ├── verify.cu                          # 与 C_ref 比较正确性
│   ├── bench.sh                           # 批量 benchmark 脚本
│   ├── gemm_common.mk                     # check / bench-run / ncu / nsys 通用目标
│   └── *.h, *.c                           # CUDA 检查、矩阵 IO 与误差比较工具
├── gemm_naive_O0/                         # 朴素 GEMM，关闭编译器优化
│   ├── Makefile
│   ├── results/                           # benchmark CSV
│   └── profiling/                         # ncu / nsys profiling 结果
├── gemm_naive/                            # 朴素 GEMM，默认 -O3 编译
│   ├── Makefile
│   ├── gemm_naive.cu
│   ├── results/
│   └── profiling/
├── gemm_shared/                           # 32×32 shared memory tiled GEMM
│   ├── Makefile
│   ├── gemm_shared.cu
│   ├── analysis/gemm_shared_lineinfo_sm_70.sass  # 带源码位置提示的 SASS 参考汇编
│   ├── results/
│   └── profiling/
└── gemm_cublas/                           # cuBLAS 对照实现
    ├── Makefile
    ├── gemm_cublas.cu
    ├── results/
    └── profiling/
```

常用入口：先在 `final/data` 下生成所需规模的输入矩阵与参考结果；随后在各实现目录执行 `make check-all` 做正确性校验，`make bench-run` 生成 benchmark CSV，`make ncu` / `make nsys` 采集 profiling。`gemm_shared` 额外提供 `make sass-lineinfo`，用于生成带源码位置提示的 SASS 参考汇编。
