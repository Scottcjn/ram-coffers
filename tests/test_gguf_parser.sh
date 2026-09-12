#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat >"$TMP/numa.h" <<'EOF'
#ifndef TEST_NUMA_H
#define TEST_NUMA_H
static inline int numa_available(void) { return 0; }
#endif
EOF

cat >"$TMP/numaif.h" <<'EOF'
#ifndef TEST_NUMAIF_H
#define TEST_NUMAIF_H
#include <stddef.h>
#define MPOL_BIND 2
#define MPOL_PREFERRED 1
#define MPOL_MF_MOVE 2
static inline int mbind(
    void *start,
    unsigned long len,
    int mode,
    const unsigned long *nodemask,
    unsigned long maxnode,
    unsigned flags
) {
    (void)start;
    (void)len;
    (void)mode;
    (void)nodemask;
    (void)maxnode;
    (void)flags;
    return 0;
}
#endif
EOF

cat >"$TMP/test_gguf_parser.c" <<'EOF'
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-coffer-mmap.h"

static void put_u32(uint8_t *buf, size_t *pos, uint32_t value) {
    memcpy(buf + *pos, &value, sizeof(value));
    *pos += sizeof(value);
}

static void put_u64(uint8_t *buf, size_t *pos, uint64_t value) {
    memcpy(buf + *pos, &value, sizeof(value));
    *pos += sizeof(value);
}

static void put_string(uint8_t *buf, size_t *pos, const char *value) {
    uint64_t len = (uint64_t)strlen(value);
    put_u64(buf, pos, len);
    memcpy(buf + *pos, value, (size_t)len);
    *pos += (size_t)len;
}

static size_t build_fixture(
    uint8_t *buf,
    size_t capacity,
    uint32_t version,
    size_t *data_offset
) {
    (void)capacity;
    size_t pos = 0;

    put_u32(buf, &pos, GGUF_MAGIC);
    put_u32(buf, &pos, version);
    put_u64(buf, &pos, 2);
    put_u64(buf, &pos, 3);

    put_string(buf, &pos, "general.alignment");
    put_u32(buf, &pos, GGUF_TYPE_UINT32);
    put_u32(buf, &pos, 32);

    put_string(buf, &pos, "general.name");
    put_u32(buf, &pos, GGUF_TYPE_STRING);
    put_string(buf, &pos, "parser-fixture");

    put_string(buf, &pos, "test.array");
    put_u32(buf, &pos, GGUF_TYPE_ARRAY);
    put_u32(buf, &pos, GGUF_TYPE_UINT32);
    put_u64(buf, &pos, 3);
    put_u32(buf, &pos, 10);
    put_u32(buf, &pos, 20);
    put_u32(buf, &pos, 30);

    put_string(buf, &pos, "blk.0.weight");
    put_u32(buf, &pos, 2);
    put_u64(buf, &pos, 2);
    put_u64(buf, &pos, 4);
    put_u32(buf, &pos, 0);
    put_u64(buf, &pos, 0);

    put_string(buf, &pos, "output.weight");
    put_u32(buf, &pos, 1);
    put_u64(buf, &pos, 8);
    put_u32(buf, &pos, 0);
    put_u64(buf, &pos, 32);

    pos = (pos + 31u) & ~31u;
    *data_offset = pos;
    memset(buf + pos, 0xA5, 64);
    return pos + 64;
}

int main(void) {
    uint8_t buffer[1024] = {0};
    size_t expected_data_offset = 0;
    size_t file_size = build_fixture(buffer, sizeof(buffer), 3, &expected_data_offset);

    coffer_mmap_ctx_t ctx = {0};
    ctx.mapped_addr = buffer;
    ctx.file_size = file_size;

    assert(coffer_parse_gguf(&ctx) == 0);
    assert(ctx.n_tensors == 2);
    assert(ctx.tensor_data_offset == expected_data_offset);

    assert(strcmp(ctx.tensors[0].name, "blk.0.weight") == 0);
    assert(ctx.tensors[0].n_dims == 2);
    assert(ctx.tensors[0].dims[0] == 2);
    assert(ctx.tensors[0].dims[1] == 4);
    assert(ctx.tensors[0].ggml_type == 0);
    assert(ctx.tensors[0].offset == 0);
    assert(ctx.tensors[0].size_bytes == 32);

    assert(strcmp(ctx.tensors[1].name, "output.weight") == 0);
    assert(ctx.tensors[1].offset == 32);
    assert(ctx.tensors[1].size_bytes == 32);

    assert(coffer_place_tensors(&ctx, 1) == 2);
    free(ctx.tensors);

    memset(buffer, 0, sizeof(buffer));
    file_size = build_fixture(buffer, sizeof(buffer), 2, &expected_data_offset);
    coffer_mmap_ctx_t v2 = {0};
    v2.mapped_addr = buffer;
    v2.file_size = file_size;
    assert(coffer_parse_gguf(&v2) == 0);
    assert(v2.header.version == 2);
    assert(v2.n_tensors == 2);
    assert(v2.tensors[0].offset == 0);
    assert(v2.tensors[0].size_bytes == 32);
    free(v2.tensors);

    coffer_mmap_ctx_t truncated = {0};
    truncated.mapped_addr = buffer;
    truncated.file_size = sizeof(gguf_header_t) + 1;
    assert(coffer_parse_gguf(&truncated) < 0);
    assert(truncated.tensors == NULL);

    puts("GGUF parser regression test: PASS");
    return 0;
}
EOF

cc -D_GNU_SOURCE -std=c11 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter \
   -I"$TMP" -I"$ROOT" "$TMP/test_gguf_parser.c" -o "$TMP/test_gguf_parser"
"$TMP/test_gguf_parser"
