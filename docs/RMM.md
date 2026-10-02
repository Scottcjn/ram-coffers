# Reduced Matrix Multiplication (RMM)

`ggml-rmm.h` implements **Reduced Matrix Multiplication**, an input-adaptive,
training-free reduction of the matrix products that dominate LLM inference.

**Source paper:** Zixuan Lan, Yanhong Li, Jiawei Zhou, *"Reduced Matrix
Multiplication: Input-Adaptive Matrix-Product Reduction for LLM Inference"*,
arXiv:2608.13426 — <https://arxiv.org/abs/2608.13426>.

## 1. The method

Every heavy Transformer op is `Y = A B`, with `A ∈ R^{n×d}` the activations
(observable at inference time) and `B ∈ R^{d×m}` the weights or the K/V cache.
Writing the product as a sum over the shared contraction axis,

```
A B = Σ_{j=1..d} A[:, j] B[j, :]
```

RMM keeps only a subset `I` of that axis:

```
RMM_ρ(A, B) = A[:, I] B[I, :],      |I| = ceil(ρ · d)
```

and chooses `I` from the *current* activations by column 2-norm:

```
s_j = ‖A[:, j]‖₂ ,   I = TopK(s, ceil(ρ · d))
```

Properties that matter for this project:

| Property | Consequence |
|---|---|
| Training-free, weight-preserving | drops into an existing GGUF/coffer model with no conversion |
| Deterministic given `A` | same input ⇒ same `I`, so runs are reproducible and attestable |
| Input-adaptive | `I` moves freely between inputs, layers, heads and decode steps |
| Minimax optimal (paper, App. E.1) | dropping `j` costs `‖A[:,j]‖₂·‖B[j,:]‖₂`; keeping the largest column norms minimises worst-case Frobenius error over every `B` consistent with the budget |
| `ρ = 1` is the identity | exact mode is bit-identical to the dense kernel |

**Proposition 1 (error bound).** With `D` the discarded index set,

```
‖A B − RMM_ρ(A, B)‖_F  ≤  Σ_{j ∈ D} ‖A[:, j]‖₂ ‖B[j, :]‖₂
```

`tests/rmm_test.c` checks this bound directly, plus the minimax claim and the
activation-aware-vs-random-selection comparison.

## 2. Why it belongs in RAM Coffers

A retention ratio on the contraction axis is a retention ratio on **rows of the
weight matrix**: `X[:, I] W[I, :]` reads only `|I|` rows of `W`. With weights
mmap'd into NUMA coffers, that is the difference between streaming a whole bank
and streaming `ρ` of it — so the arithmetic saving arrives as a **bandwidth and
page-residency saving**, which is the axis a POWER8 node is actually short of.
`rmm_prefetch_rows()` issues `dcbt` (POWER) or `__builtin_prefetch` for exactly
the retained rows, so at `ρ = 0.5` half the bank never enters cache.

Relation to the existing collapse headers: `ggml-topk-collapse-vsx.h` prunes
attention *scores* after `QKᵀ` has already been paid for. RMM shrinks the
products themselves — the head dimension inside `QKᵀ`, and optionally the token
axis inside `PV` — so the saving is in the multiply, not in the softmax. The two
are composable.

## 3. API reference

All functions are `static inline` in `ggml-rmm.h`; matrices are row-major
`float`.

### Configuration

```c
typedef struct {
    float rho_d;     /* retention on the feature/contraction axis */
    float rho_t;     /* retention on the token axis of PV; 1.0f = dense PV */
    int   min_keep;  /* never reduce below this many indices */
} rmm_config_t;

rmm_config_t cfg = rmm_config_default();   /* 0.75 / 1.0 / 8 */
```

Compile-time defaults: `RMM_RHO_D_DEFAULT`, `RMM_RHO_T_DEFAULT`,
`RMM_MIN_KEEP`. The paper's component analysis finds attention products far more
reducible than MLP ones, so give MLPs a higher `rho_d` than attention.

`int rmm_keep_count(int d, float rho, int min_keep)` returns
`ceil(rho·d)` clamped to `[min(min_keep, d), d]`.

### Workspace

Selection buffers are sized once per model, never per call — these paths run
inside the decode loop.

```c
rmm_workspace_t ws;
/* max_probs = max_q_rows * max_tokens, or 0 to leave PV dense */
if (!rmm_workspace_init(&ws, max_d, max_tokens, max_probs)) { /* OOM */ }
...
rmm_workspace_free(&ws);
```

### Selection

```c
void  rmm_feature_scores(const float *A, int n, int d, int lda, float *scores);
float rmm_kth_largest   (const float *scores, int d, int k, float *scratch);
int   rmm_select_topk   (const float *scores, int d, int k,
                         int32_t *idx, float *scratch);
```

`rmm_select_topk()` quickselects the threshold, then sweeps `j` ascending taking
everything above it and as many ties as the budget allows. The ascending sweep
is what makes selection reproducible — ties break to the **lowest index**, not
to quickselect's pivot order — and it leaves `idx` sorted, so the gathers in the
inner loops move forwards through memory.

### Reduced products

```c
void rmm_gemv_reduced(const float *W, const float *x, float *y,
                      int rows, int d, int ldw, const int32_t *idx, int k);

void rmm_gemm_reduced(const float *A, const float *B, float *Y,
                      int n, int d, int m, int lda, int ldb, int ldy,
                      const int32_t *idx, int k);

void rmm_prefetch_rows(const float *B, int ldb, const int32_t *idx, int k);
```

### Projections (paper §3.3)

```c
int rmm_project     (const float *X, int L, int d, const float *W, int dout,
                     float *Y, const rmm_config_t *cfg, rmm_workspace_t *ws);

int rmm_project_gemv(const float *x, int d, const float *W, int rows,
                     float *y, const rmm_config_t *cfg, rmm_workspace_t *ws);
```

One call covers Q/K/V projections, the attention output projection, and an FFN's
gate/up/down matrices. `rmm_project_gemv()` is the decode-step form for weights
stored row-major as `rows × d` — the layout the coffer headers mmap. Both return
the retained count `k`, or `-1` if the workspace is too small.

### Attention (paper §3.3)

```c
int rmm_attention(float *out, const float *Q, const float *K, const float *V,
                  int Lq, int Lk, int dh, int causal,
                  const rmm_config_t *cfg, rmm_workspace_t *ws);
```

Per head: score the head dimension with `s_j = ‖Q[:, j]‖₂`, reduce `QKᵀ` over
the retained dimensions, softmax (causal-masked if asked), then optionally
reduce `PV` over the token axis using `a_t = ‖P[:, t]‖₂`.

- **The scale stays `1/sqrt(dh)`**, not `1/sqrt(k)`: RMM approximates the dense
  logits, so rescaling by the reduced width would shift the softmax temperature
  away from the model's calibration.
- **Grouped-query attention:** selection is per query head on `Q`, with the
  retained dimensions gathered from the shared K/V — call once per query head,
  passing that head's `Q` with the K/V of its group.
- **Token reduction drops the discarded attention mass** instead of
  renormalising, exactly as the paper defines `PV → P[:, T] V[T, :]`. Output
  rows are therefore slightly shrunk; that is part of the approximation being
  measured, not a bug.
- Token reduction (`rho_t < 1`) requires the `probs` buffer, i.e. a non-zero
  `max_probs` at init, because `a_t` is taken over the whole query block and
  cannot be formed while streaming `P` row by row. Without it the call returns
  `-1`.

### Dense references and statistics

```c
void rmm_gemv_dense     (const float *W, const float *x, float *y,
                         int rows, int d, int ldw);
void rmm_attention_dense(float *out, const float *Q, const float *K,
                         const float *V, int Lq, int Lk, int dh,
                         int causal, float *scores);

void rmm_stats_reset (void);
void rmm_stats_report(void);   /* selections, and MACs kept vs dense */
```

Statistics report the *realised* reduction rather than the requested one —
`min_keep` and causal masking make the two differ.

## 4. Usage sketch

```c
#include "ggml-rmm.h"

rmm_config_t cfg = rmm_config_default();
cfg.rho_d = 0.5f;                    /* attention: half the head dimension */

rmm_workspace_t ws;
rmm_workspace_init(&ws, /*max_d*/ 4096, /*max_tokens*/ 8192, /*max_probs*/ 0);

/* decode-step projection out of a coffer-resident weight bank */
rmm_project_gemv(x, d, W_bank, rows, y, &cfg, &ws);

/* one attention head, causal */
rmm_attention(out, Q, K, V, Lq, Lk, dh, /*causal*/ 1, &cfg, &ws);

rmm_stats_report();
rmm_workspace_free(&ws);
```

Set `cfg.rho_d = cfg.rho_t = 1.0f` to fall back to exact dense behaviour without
removing the call.

## 5. Portability

- **Scalar C11** everywhere; this is the path CI exercises on x86-64.
- **VSX/AltiVec** fast path under `#if defined(__VSX__)`, using `vec_xl` /
  `vec_xst` / `vec_madd` via `power8-compat.h`. Plain AltiVec `vec_ld` is
  deliberately not used: these loops load at arbitrary float offsets, which
  `vec_ld` would silently truncate to a 16-byte boundary.
- `dcbt` prefetch on `__powerpc__` / `__powerpc64__`, `__builtin_prefetch`
  elsewhere.

## 6. Build and test

```bash
make            # build and run tests/rmm_test.c (scalar path)
make power8     # build with -mcpu=power8 -maltivec -mvsx (VSX path)
make clean
```

The test suite covers: retention arithmetic, TopK correctness and determinism
(including tie resolution), column-norm scoring, bit-exact dense equivalence at
`ρ = 1` (GEMV and attention), the Proposition 1 bound, activation-aware vs
random selection at equal budget, the minimax claim, attention accuracy at
`ρ_d = 0.5`, token reduction and its MAC accounting, workspace refusal on
undersized buffers, and a dense-vs-reduced GEMV benchmark.

Reference results, 4096×4096 GEMV, `ρ_d = 0.5`, selection included in the timing:

| Host | Dense | Reduced | Speed-up |
|---|---|---|---|
| x86-64 (scalar path) | 21.87 ms | 12.53 ms | 1.75× |
| POWER8 VSX path (qemu-ppc64le, syntax/behaviour check only) | 2463.51 ms | 870.39 ms | 2.83× |

The POWER8 row is emulated, so treat the ratio — not the absolute times — as the
signal; it needs re-measuring on real POWER8 hardware.

## 7. Citation

```bibtex
@misc{lan2026rmm,
  title  = {Reduced Matrix Multiplication: Input-Adaptive Matrix-Product
            Reduction for LLM Inference},
  author = {Lan, Zixuan and Li, Yanhong and Zhou, Jiawei},
  year   = {2026},
  eprint = {2608.13426},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2608.13426}
}
```
