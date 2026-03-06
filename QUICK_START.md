# RAM Coffers Quick Start

Get NUMA-aware weight banking running on your POWER8 system in 5 minutes.

## Prerequisites

- IBM POWER8 or later with multiple NUMA nodes
- GCC 10+ with `-mcpu=power8 -mvsx -maltivec`
- llama.cpp source tree

## 1. Copy Headers

```bash
cp ggml-ram-coffers.h ~/llama.cpp/ggml/src/ggml-cpu/arch/powerpc/
cp ggml-coffer-mmap.h ~/llama.cpp/ggml/src/ggml-cpu/arch/powerpc/
cp ggml-neuromorphic-coffers.h ~/llama.cpp/ggml/src/ggml-cpu/arch/powerpc/
```

## 2. Build with Coffers Enabled

```bash
cd ~/llama.cpp
mkdir build-coffers && cd build-coffers
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3" \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3"
make -j32
```

## 3. Run with NUMA Interleave

```bash
# For models under 100GB (single NUMA node)
numactl --cpunodebind=1 --membind=1 ./bin/llama-cli -m model.gguf -p "Hello" -t 32

# For models over 100GB (interleave across all nodes)
numactl --interleave=all ./bin/llama-cli -m model.gguf -p "Hello" -t 64
```

## 4. Verify Coffers Are Active

Look for this banner in output:
```
PSE Vec_Perm Collapse Active - POWER8 S824
 - RAM Coffers: ENABLED | NUMA-aware weight banking
```

## Performance Reference

| Model | Without Coffers | With Coffers | Speedup |
|-------|----------------|--------------|---------|
| TinyLlama 1.1B Q4 | 84 t/s | 147 t/s | 1.75x |

See [README.md](README.md) for full documentation.
