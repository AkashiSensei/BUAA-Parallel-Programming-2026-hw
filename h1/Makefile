# macOS (Apple clang + Homebrew libomp)
LIBOMP_PREFIX ?= $(shell [ -d /opt/homebrew/opt/libomp ] && echo /opt/homebrew/opt/libomp || echo /usr/local/opt/libomp)

CC       = clang
CPPFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
CFLAGS   = -O3 -Wall -Wextra
LDFLAGS  = -L$(LIBOMP_PREFIX)/lib -lomp -lm

TARGETS = matmul_omp sin_taylor_omp

.PHONY: all clean run-demo

all: $(TARGETS)

matmul_omp: matmul_omp.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ matmul_omp.c $(LDFLAGS)

sin_taylor_omp: sin_taylor_omp.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ sin_taylor_omp.c $(LDFLAGS)

clean:
	rm -f $(TARGETS)
