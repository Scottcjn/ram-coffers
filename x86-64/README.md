# x86-64 PSE — Non-Bijunctive Collapse via AES-NI

Third port of the collapse, after POWER8 (`ggml-vcipher-collapse.h`) and ARM
(`apple-silicon/aes-entropy-collapse.h`).

```
make bench
```

## Architecture mapping

| POWER8 | x86-64 | Purpose |
|--------|--------|---------|
| `vcipher` | `_mm_aesenc_si128` | AES round — **exactly the same function** |
| `vcipherlast` | `_mm_aesenclast_si128` | AES round without MixColumns |
| `vec_perm(a,a,p)` | `_mm_shuffle_epi8` | Single-source byte permute |
| `vec_perm(a,b,p)` | two `pshufb` + `blendv` | Dual-source permute (not needed here) |
| `vec_cmpgt` / `vec_sel` | `_mm_cmpgt_ps` / `_mm_blendv_ps` | Threshold and select |
| `mftb` | `rdtsc` | Hardware counter (off by default, see below) |

## x86 is the exact port; ARM is not

`aesenc` and `vcipher` both compute

```
MixColumns(ShiftRows(SubBytes(state))) XOR key
```

so every mode transcribes literally, and `bench-aes-collapse.c` checks it
against a scalar AES round written from FIPS-197 rather than against another
intrinsic.

ARM is the odd one out. `AESE` applies the key *first* and `AESMC` adds no
final XOR, so `vaeseq_u8` + `vaesmcq_u8` computes `aesenc(state XOR key, 0)` —
a different function of `(state, key)`, agreeing only when the key is zero.
The diffusion is just as good, it is still a full AES round, but the three
ports do not produce the same bytes for the same inputs, so a pattern
generated on POWER8 cannot be reproduced on an M2. `x86_aes_arm_order()`
exists for when you need to match the Apple port instead.

The mapping table in `apple-silicon/PAPER-architecture-general-pse.md`
(Appendix C) lists the ARM pair as equivalent to `vcipher`. It isn't; two of
the checks in the bench demonstrate that.

## Two deliberate divergences from POWER8

**Round keys are deterministic by default.** The POWER8 header reseeds from
`mftb` on every call, so no two runs produce the same patterns. That suits
entropy injection and rules out everything that has to be reproducible.
Here the key comes from `(layer, position, counter)`; build with
`-DX86_AES_COLLAPSE_ENTROPY=1` for the original behaviour.

**Byte order.** `vec_perm` numbers bytes from the most significant end on
big-endian POWER, `pshufb` from the least significant, so patterns are
mirrored between the two. `x86_aes_beswap()` converts.

## Cost

Measured by `make bench` on a Xeon Platinum 8559C (Emerald Rapids):

| | ns per AES round |
|---|---|
| Dependent chain (latency bound) | 0.75 |
| Four chains in flight (throughput bound) | 0.19 — 5.2 G rounds/s |
| `pshufb`, dependent, for scale | 0.27 |

So a round costs about three `pshufb` latencies and is free on throughput,
which is the same argument the POWER8 header makes. Eight independent chains
saturate the unit; `x86_aes_collapse_8way` is built around that.

Numbers from one machine. Zen 2 has AES-NI but no VAES, and its own latency
and port count, so it will differ — remeasure before quoting.

## Mode 4 does not work

`x86_aes_attention_score` is ported for parity and is **not** a similarity
metric, despite what the POWER8 comment claims for it. Over 200k Q/K pairs its
correlation with cosine similarity is **+0.002**, and the score for `Q == K`
sits in the middle of the range it produces for unrelated pairs.

The avalanche property is the cause: it is what the mode is built on and it is
exactly what destroys any monotonic relationship between input distance and
output byte-sum. A near-zero XOR is not a near-zero round output, because
SubBytes maps `0x00` to `0x63`.

What it is: a fast deterministic 12-bit hash of `Q XOR K`, which detects
equality and nothing weaker. Kept and tested as such rather than deleted,
because the other two ports ship it and a silent divergence would be worse
than a loud caveat.

## Not used by gen9-cluster

The DeepSeek V4 Pro fleet in `gen9-cluster/` does not call any of this, on
purpose:

- Its attention runs GPU-side on RDNA2 through Vulkan/RADV, and RDNA2 has no
  AES instruction. A collapse there means pinning attention to the CPU or
  doing S-box lookups in GLSL, either of which costs more than it saves.
- Every node in that fleet — PS5, Series X/S, 4700S, 4800S, BC-250 — is Zen 2,
  so AES-NI is universal and VAES is unavailable. Only a Steam Machine class
  box would get the four-wide rounds.
- `G9XC` makes exact per-expert reduction the default. Anything that injects
  entropy into the numerics belongs behind the `FAST` opt-in, not in the
  default path.

This directory is the portable primitive. Wiring it into a model is a separate
decision.
