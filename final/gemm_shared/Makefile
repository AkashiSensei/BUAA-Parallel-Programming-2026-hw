KERNEL_SRC = gemm_shared.cu

# 32×32 tile：2×32²×8 B = 16 KiB 共享内存/块（V100 每块上限 48 KiB）
BLOCK = 32

include ../util/gemm_common.mk
