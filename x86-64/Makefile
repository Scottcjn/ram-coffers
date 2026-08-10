# Elyan Labs — x86-64 AES-NI collapse
#
#   make        build and nothing else
#   make bench  build and run the checks and the benchmark
#
# AES-NI is the only hard requirement. SSE4.1 is for blendv and ptest in the
# hybrid collapse and the tests; every AES-NI part has it.

CC ?= cc
CFLAGS ?= -O2 -maes -msse4.1 -Wall -Wextra
LDFLAGS = -lm

# -march=native picks up VAES on Zen 3 / Ice Lake and later. It is deliberately
# not the default: the fleet this lives next to is Zen 2, which has AES-NI and
# no VAES, and a binary built here must still run there.

.PHONY: all bench clean

all: bench-aes-collapse

bench-aes-collapse: bench-aes-collapse.c aes-collapse.h
	$(CC) $(CFLAGS) -I. -o $@ bench-aes-collapse.c $(LDFLAGS)

bench: bench-aes-collapse
	./bench-aes-collapse

clean:
	rm -f bench-aes-collapse
