# 与 gemm_naive 相同源码，编译选项关闭优化（用于与 -O3 对比实验）。
KERNEL_SRC = ../gemm_naive/gemm_naive.cu
IMPL_NAME = gemm_naive_O0
NCU_KERNEL = gemm_naive_kernel

include ../util/gemm_common.mk

NVCCFLAGS = -O0 -std=c++11 -Xcompiler -Wall,-Wextra -Xptxas -O0 --fmad=false
