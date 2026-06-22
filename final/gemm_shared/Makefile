KERNEL_SRC = gemm_shared.cu

# 32×32 tile：2×32²×8 B = 16 KiB 共享内存/块（V100 每块上限 48 KiB）
BLOCK = 32

include ../util/gemm_common.mk

# Generate a line-info annotated SASS reference for reading and report analysis.
SASS_ARCH ?= sm_70
SASS_DIR ?= analysis
SASS_NVCC ?= /usr/local/cuda-12.4/bin/nvcc
NVDISASM ?= /usr/local/cuda-12.4/bin/nvdisasm
SASS_FILE ?= $(SASS_DIR)/gemm_shared_lineinfo_$(SASS_ARCH).sass
SASS_CUBIN ?= /tmp/gemm_shared_lineinfo_$(SASS_ARCH).cubin

.PHONY: sass-lineinfo
sass-lineinfo:
	@mkdir -p $(SASS_DIR)
	$(SASS_NVCC) $(NVCCFLAGS) -lineinfo -arch=$(SASS_ARCH) $(INCLUDES) -cubin $(KERNEL_SRC) -o $(SASS_CUBIN)
	$(NVDISASM) --print-line-info --print-code $(SASS_CUBIN) > $(SASS_FILE)
