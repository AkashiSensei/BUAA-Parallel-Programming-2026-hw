CC = gcc
NVCC = nvcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -I../util
NVCCFLAGS = -O3 -std=c++11 -Xcompiler -Wall,-Wextra -I../util
CUBLAS_LDFLAGS = -lcublas

UTIL_SRC = ../util/mat_io.c

GEN_DATA = gen_data
GEN_CREF = gen_cref
SIZES = 512 1024 2048 4096 8192

.PHONY: all generate gen-cref clean distclean

all: $(GEN_DATA) $(GEN_CREF)

$(GEN_DATA): gen_data.c ../util/mat_data.h
	$(CC) $(CFLAGS) -o $@ gen_data.c

$(GEN_CREF): gen_cref.cu $(UTIL_SRC) ../util/mat_io.h ../util/mat_data.h ../util/cuda_check.h ../util/cublas_check.h
	$(NVCC) $(NVCCFLAGS) -o $@ gen_cref.cu $(UTIL_SRC) $(CUBLAS_LDFLAGS)

generate: $(GEN_DATA)
	./$(GEN_DATA) $(SIZES)

gen-cref: $(GEN_CREF)
	CUDA_VISIBLE_DEVICES=0 ./$(GEN_CREF) $(SIZES)

clean:
	rm -f $(GEN_DATA) $(GEN_CREF)

distclean: clean
	rm -f N*.bin
