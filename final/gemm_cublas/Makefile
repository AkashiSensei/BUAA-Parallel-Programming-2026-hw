KERNEL_SRC = gemm_cublas.cu
IMPL_NAME = gemm_cublas
NVCC ?= /usr/local/cuda-12.4/bin/nvcc

# cuBLAS ignores the benchmark block shape, but keep the argument valid for the
# shared harness. Nsight Compute profiles the internal library kernel.
BLOCK = 1
LDLIBS = -lcublas
NCU_KERNEL = regex:.*

include ../util/gemm_common.mk
