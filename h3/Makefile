MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -lm

TARGET := matmul_mpi
SRC := matmul_mpi.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS) -lm

clean:
	rm -f $(TARGET)
