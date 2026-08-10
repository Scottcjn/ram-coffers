# Kernels for the gen9 cluster.
#
# The CPU path builds everywhere and is the only one the test suite requires.
# The GPU paths build only where their toolchain exists, and their absence is a
# supported configuration rather than a failure: a fleet may legitimately be all
# CPU (4700S boards, Xbox Dev Mode, PS5 HEN).
#
#   make            CPU shared library (+ the self-test binary)
#   make test       build and run the kernel self-test
#   make vulkan     compile the compute shader to SPIR-V (needs glslangValidator)
#   make hip        HIP library for gfx1013 (needs hipcc; see expert_hip.hip)
#   make all-avail  everything whose toolchain is present

CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -fPIC
LDFLAGS ?= -lm

# -march=native is wrong here: the build host is usually not the console. Zen 2
# is the common denominator across PS5, Series X/S and the salvage boards.
ARCH_FLAGS ?= -march=znver2 -mtune=znver2
FALLBACK_ARCH_FLAGS := -mavx2 -mfma

OPENMP := $(shell $(CC) -fopenmp -E - </dev/null >/dev/null 2>&1 && echo -fopenmp)
ARCH_OK := $(shell $(CC) $(ARCH_FLAGS) -E - </dev/null >/dev/null 2>&1 && echo yes)
ifneq ($(ARCH_OK),yes)
ARCH_FLAGS := $(FALLBACK_ARCH_FLAGS)
endif

LIB     := libgen9_cpu.so
OBJS    := fp8.o expert_avx2.o
SPV     := expert.comp.spv
HIPLIB  := libgen9_hip.so

GLSLC       := $(shell command -v glslangValidator 2>/dev/null)
HIPCC       := $(shell command -v hipcc 2>/dev/null)

.PHONY: all test vulkan hip all-avail clean

all: $(LIB)

%.o: %.c fp8.h
	$(CC) $(CFLAGS) $(ARCH_FLAGS) $(OPENMP) -c $< -o $@

$(LIB): $(OBJS)
	$(CC) -shared $(OPENMP) $(OBJS) -o $@ $(LDFLAGS)

kernel_test: kernel_test.c $(OBJS)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) $(OPENMP) $^ -o $@ $(LDFLAGS)

test: kernel_test
	./kernel_test

vulkan: $(SPV)

$(SPV): expert.comp
ifeq ($(GLSLC),)
	@echo "glslangValidator not found; skipping the Vulkan shader."
	@echo "On Debian/Ubuntu: apt install glslang-tools"
else
	$(GLSLC) -V --target-env vulkan1.1 -S comp $< -o $@
endif

hip:
ifeq ($(HIPCC),)
	@echo "hipcc not found; skipping the HIP kernel (Vulkan is the default path)."
else
	$(HIPCC) -O3 --offload-arch=gfx1013 -fPIC -shared expert_hip.hip -o $(HIPLIB)
endif

all-avail: all vulkan hip

clean:
	rm -f $(OBJS) $(LIB) $(SPV) $(HIPLIB) kernel_test
