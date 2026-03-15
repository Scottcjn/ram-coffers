# Elyan Labs — Apple Silicon PSE Build
#
# On Mac:   make
# On Linux: make CC=gcc CFLAGS="-O3 -march=armv8-a+crypto"
# Cross:    make CC=aarch64-linux-gnu-gcc

CC ?= clang
CFLAGS ?= -O3 -march=armv8-a+crypto -Wall -Wextra
LDFLAGS = -lm

# Detect Apple Silicon
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	CFLAGS += -DAPPLE_SILICON_NATIVE=1
	$(info Building for Apple Silicon (native))
endif
endif

.PHONY: all clean bench

all: bench-pse

bench-pse: bench-pse-apple.c apple-pse-config.h neon-collapse.h aes-entropy-collapse.h unified-memory-coffers.h apple-pse-integration.h
	$(CC) $(CFLAGS) -I. -o $@ bench-pse-apple.c $(LDFLAGS)

bench: bench-pse
	./bench-pse

clean:
	rm -f bench-pse
