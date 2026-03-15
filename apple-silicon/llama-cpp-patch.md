# llama.cpp Integration for Apple Silicon PSE

## Quick Start on Mac Mini M2

```bash
# 1. Enable Remote Login: System Settings → General → Sharing → Remote Login

# 2. Clone and build
git clone https://github.com/Scottcjn/ram-coffers.git ~/ram-coffers
cd ~/ram-coffers/apple-silicon
make bench    # Run standalone benchmark first

# 3. Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git ~/llama-pse
cd ~/llama-pse

# 4. Copy PSE headers
cp ~/ram-coffers/apple-silicon/*.h ggml/src/ggml-cpu/

# 5. Build with PSE
mkdir build-pse && cd build-pse
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-DPSE_ENABLED=1" \
    -DCMAKE_CXX_FLAGS="-DPSE_ENABLED=1"
make -j8

# 6. Run benchmark
./bin/llama-bench -m ~/models/tinyllama-1.1b-q4.gguf -t 4
```

## Integration Points in llama.cpp

### 1. Attention Collapse (ggml-cpu/ops.cpp)

In `ggml_compute_forward_flash_attn_ext_f16_one_chunk()`, after computing
Q·K^T scores but before softmax:

```cpp
#ifdef PSE_ENABLED
#include "apple-pse-integration.h"

// After: float S[KQ_NHEAD] = { ... Q·K^T scores ... }
// Before: softmax(S)
pse_apple_collapse_attention(S, KQ_NHEAD, il, ih, pos);
#endif
```

### 2. Entropy Burst (llama-sampling.cpp)

In the sampling chain, after logits are computed:

```cpp
#ifdef PSE_ENABLED
#include "apple-pse-integration.h"

// In llama_sampler_apply()
pse_apple_entropy_burst(logits, n_vocab);
#endif
```

### 3. Model Load (llama-model.cpp)

During model loading, initialize coffers:

```cpp
#ifdef PSE_ENABLED
#include "apple-pse-integration.h"

// After model weights are loaded
pse_apple_init(model->data, model->size, model->n_layers, layer_size);
#endif
```
