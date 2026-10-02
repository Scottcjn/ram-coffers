# RAM Coffers is a header-only library, so there is nothing to link into a
# product here; what this Makefile builds is the portable self-tests, which is
# what CI needs to be able to run on an x86 runner.
#
#   make            -> build and run the portable tests (default)
#   make rmm-test   -> build tests/rmm_test.c only
#   make power8     -> compile-check the POWER8/VSX headers (needs a POWER8
#                      target compiler; skipped elsewhere)

BUILD_DIR ?= build
CC        ?= cc
# _POSIX_C_SOURCE: strict c11 hides clock_gettime, which the tests time with.
CFLAGS    ?= -O2 -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
LDLIBS    ?= -lm

.PHONY: all test rmm-test power8 clean

all: test

test: $(BUILD_DIR)/rmm_test
	$(BUILD_DIR)/rmm_test

rmm-test: $(BUILD_DIR)/rmm_test

$(BUILD_DIR)/rmm_test: tests/rmm_test.c ggml-rmm.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ tests/rmm_test.c $(LDLIBS)

# The VSX path of ggml-rmm.h only exists on POWER8; this target exercises it.
POWER8_CFLAGS ?= $(CFLAGS) -mcpu=power8 -maltivec -mvsx

power8: $(BUILD_DIR)/rmm_test_power8

$(BUILD_DIR)/rmm_test_power8: tests/rmm_test.c ggml-rmm.h power8-compat.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(POWER8_CFLAGS) -o $@ tests/rmm_test.c $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)
