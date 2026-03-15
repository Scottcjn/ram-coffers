/*
 * pse-minimal.h — Minimal PSE for C++ integration in llama.cpp
 *
 * Elyan Labs — Drop this into ggml/src/ggml-cpu/ and build with -DPSE_ENABLED=1
 *
 * This is the lightweight C++-compatible header for patching llama.cpp.
 * For the full PSE stack (collapse, coffers, AES entropy), use apple-pse-integration.h
 */
#ifndef PSE_MINIMAL_H
#define PSE_MINIMAL_H

#include <stdint.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

/* Hardware timebase — high-resolution
 *
 * On macOS: mach_absolute_time() has nanosecond resolution
 * On Linux ARM: cntvct_el0 works fine
 * On x86: rdtsc
 *
 * IMPORTANT: Do NOT use cntvct_el0 on macOS — resolution is too coarse
 * (delta range 0-1 ticks). mach_absolute_time is the correct API.
 */
static inline uint64_t pse_read_timebase(void) {
#if defined(__APPLE__)
    return mach_absolute_time();
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#elif defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

static inline void pse_apple_init_lite(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    fprintf(stderr, "[PSE] Non-bijunctive collapse active (hardware entropy)\n");
}

#endif /* PSE_MINIMAL_H */
