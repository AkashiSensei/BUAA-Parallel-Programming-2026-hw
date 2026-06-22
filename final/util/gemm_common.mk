# 各 gemm_* 实现目录通过 include 引入本文件。
# 需定义 KERNEL_SRC（如 gemm_naive.cu），可选 IMPL_NAME（默认取目录名）。

NVCC ?= nvcc
UTIL_DIR ?= ../util
DATA_DIR ?= ../data
CUDA_DEVICE ?= 0

ifndef KERNEL_SRC
$(error 请在 Makefile 中定义 KERNEL_SRC)
endif

IMPL_NAME ?= $(notdir $(patsubst %/,%,$(CURDIR)))

# 线程块边长（BLOCK×BLOCK）；朴素实现默认 16，共享内存等实现可在 include 前设为 32。
BLOCK ?= 16

NVCCFLAGS = -O3 -std=c++11 -Xcompiler -Wall,-Wextra
INCLUDES = -I$(UTIL_DIR) -I.
LDLIBS ?=
UTIL_BENCH = $(UTIL_DIR)/bench.cu
UTIL_VERIFY = $(UTIL_DIR)/verify.cu
UTIL_IO = $(UTIL_DIR)/mat_io.c
UTIL_COMPARE = $(UTIL_DIR)/mat_compare.c

BENCH = bench
VERIFY_BIN = verify_gemm
SIZES = 512 1024 2048 4096 8192
RESULTS_DIR ?= results
# 默认 PATH 中的 ncu 会指向 2026.x，已不再支持 Volta（V100/gv100）。
# CUDA 12.6 节点请使用仍含 gv100 的 Nsight Compute 2024.3.2。
NCU ?= /opt/nvidia/nsight-compute/2024.3.2/ncu
# ncu 2024.3.2 可用 section set：basic / detailed / full（无 speed）。
# basic 仅 4 个 section；detailed 含 Memory/Compute Workload Analysis；full 再含指令/调度等。
NCU_SET ?= full
# 跳过 warmup 那次 launch，只 profile 计时段的 1 次 kernel（见 bench.cu）。
NCU_LAUNCH_SKIP ?= 1
NCU_LAUNCH_COUNT ?= 1
NCU_KERNEL ?= $(IMPL_NAME)_kernel
ifneq ($(strip $(NCU_KERNEL)),)
NCU_KERNEL_ARG = -k $(NCU_KERNEL)
endif
NCU_ARGS = --set $(NCU_SET) --launch-skip $(NCU_LAUNCH_SKIP) --launch-count $(NCU_LAUNCH_COUNT) \
	$(NCU_KERNEL_ARG) --cache-control all
# 默认 /usr/local/cuda/bin/nsys 未正确安装；独立安装于 /opt/nvidia/nsight-systems/。
NSYS ?= /opt/nvidia/nsight-systems/2024.2.3/bin/nsys
# nsys 无 ncu 式 section set；默认 trace 已含 cuda/nvtx/osrt，但 GUI 需显式开启部分项。
# dell-65 上 kernel paranoid=4，CPU 采样/上下文切换不可用（见 nsys status --environment）。
NSYS_TRACE ?= cuda,nvtx,osrt
NSYS_CUDA_MEM ?= true
NSYS_STATS ?= true
# profiling 时间线只保留 1 次 kernel launch（bench 无 launch-skip 机制）。
NSYS_WARMUP ?= 0
NSYS_ITERS ?= 1
NSYS_ARGS = profile --force-overwrite=true \
	--trace=$(NSYS_TRACE) --cuda-memory-usage=$(NSYS_CUDA_MEM) --stats=$(NSYS_STATS)
PROFILING_DIR ?= profiling
NCU_DIR ?= $(PROFILING_DIR)
PROFILING_SIZES = 1024 2048 4096 8192

.PHONY: all check check-all bench-run clean run verify \
	ncu nsys $(addprefix ncu-,$(PROFILING_SIZES)) $(addprefix nsys-,$(PROFILING_SIZES))

verify: check

all: $(BENCH) $(VERIFY_BIN)

$(BENCH): $(UTIL_BENCH) $(KERNEL_SRC) $(UTIL_IO) $(UTIL_DIR)/gemm_launch.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(UTIL_BENCH) $(KERNEL_SRC) $(UTIL_IO) $(LDLIBS)

$(VERIFY_BIN): $(UTIL_VERIFY) $(KERNEL_SRC) $(UTIL_IO) $(UTIL_COMPARE) $(UTIL_DIR)/gemm_launch.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(UTIL_VERIFY) $(KERNEL_SRC) $(UTIL_IO) $(UTIL_COMPARE) $(LDLIBS)

run: $(BENCH)
ifndef N
	$(error 请指定矩阵规模，例如: make run N=1024)
endif
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ./$(BENCH) -n $(N) --data-dir $(DATA_DIR) --block $(BLOCK)

check: $(VERIFY_BIN)
ifndef N
	$(error 请指定矩阵规模，例如: make check N=1024)
endif
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ./$(VERIFY_BIN) -n $(N) --data-dir $(DATA_DIR) --block $(BLOCK)

check-all: $(VERIFY_BIN)
	@for n in $(SIZES); do \
		CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ./$(VERIFY_BIN) -n $$n --data-dir $(DATA_DIR) --block $(BLOCK) || exit 1; \
	done

bench-run: $(BENCH)
	GEMM_DIR=$(CURDIR) IMPL_NAME=$(IMPL_NAME) DATA_DIR=$(DATA_DIR) \
		RESULTS_DIR=$(RESULTS_DIR) CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) BLOCK=$(BLOCK) \
		bash $(UTIL_DIR)/bench.sh

define ncu-target
ncu-$(1): $(BENCH)
	@mkdir -p $(NCU_DIR)
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(NCU) $(NCU_ARGS) --force-overwrite \
		-o $(NCU_DIR)/$(IMPL_NAME)_N$(1) \
		./$(BENCH) -n $(1) --data-dir $(DATA_DIR) --impl $(IMPL_NAME) \
		--block $(BLOCK) --warmup 1 --iters 1
endef
$(foreach n,$(PROFILING_SIZES),$(eval $(call ncu-target,$(n))))

ncu: $(addprefix ncu-,$(PROFILING_SIZES))

define nsys-target
nsys-$(1): $(BENCH)
	@mkdir -p $(PROFILING_DIR)
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(NSYS) $(NSYS_ARGS) \
		-o $(PROFILING_DIR)/$(IMPL_NAME)_N$(1) \
		./$(BENCH) -n $(1) --data-dir $(DATA_DIR) --impl $(IMPL_NAME) \
		--block $(BLOCK) --warmup $(NSYS_WARMUP) --iters $(NSYS_ITERS)
endef
$(foreach n,$(PROFILING_SIZES),$(eval $(call nsys-target,$(n))))

nsys: $(addprefix nsys-,$(PROFILING_SIZES))

clean:
	rm -f $(BENCH) $(VERIFY_BIN)
