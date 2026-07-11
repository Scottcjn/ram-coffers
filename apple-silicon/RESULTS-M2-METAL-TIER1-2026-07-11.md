# Tier-1 Metal Collapse: Prune+Amplify Alone Reproduces the Behavioral Shift (M2, 2026-07-11)

Follow-up to `RESULTS-M2-2026-07-09.md` (NEON standalone) and the CPU-hook A/B.
Research question: **does the PSE collapse behavioral divergence require the
`vec_perm`/`TBL` permutation, or is deterministic prune+amplify sufficient?**

Hardware: Mac Mini M2, Apple clang 17, llama.cpp Metal backend (GPU path).
Model: tinyllama-1.1b-chat-q4km. Constants: PSE_PRUNE_THRESH=0.7, PSE_AMPLIFY=1.20.

## What was built
A "tier-1" collapse in the **Metal** softmax kernel (`kernel_soft_max` and
`kernel_soft_max_4` in `ggml-metal.metal`): for each pre-softmax score
`v = psrc0[i]*scale + slope*mask`, when `GGML_METAL_COLLAPSE=1`, apply
`v = (fabs(v) < 0.7) ? 0 : v*1.20` before the max/exp passes. NO permutation,
NO entropy. Flag plumbed via a `float collapse` field in the softmax kargs,
set from the env var in `ggml-metal-ops.cpp`. (Edits uncommitted on the M2,
`.bak` backups kept.)

## THE ANSWER: yes, behavior survives without the permute

GPU (`-ngl 99 -fa off`), greedy temp-0, seed 42, char-for-char DIFFERENT and
byte-reproducible across repeats:

| | output |
|---|---|
| OFF | "The meaning of life is a complex and multifaceted concept that has been the subject of much debate…" |
| ON | "1. To live a fulfilling and meaningful life is a central concept in many cultures and relig…" |

Confirmed on additional prompts. So the **non-bijunctive permutation was NOT
the source of the behavioral change** — it was a CPU/POWER8-SIMD implementation
detail (a cheap way to do prune+amplify in one instruction). Constraint-bound
selection (prune weak + amplify strong) is the active ingredient, and it is
portable to any GPU/ISA as a trivial per-element op.

## Cost on Metal: FREE

`llama-bench -ngl 99 -fa 0 -t 4 -p 512 -n 32`:

| | pp512 t/s | tg32 t/s |
|---|---|---|
| OFF | 1131.30 ± 1.98 | 103.73 ± 0.29 |
| ON  | 1130.49 ± 2.51 | 103.74 ± 0.24 |

Zero cost within noise. This **flips the CPU-ARM finding**: the ~1.5% persistent
tax measured on CPU (see the 7B context sweep) is just CPU being slow at
per-element work; the GPU does it for nothing. On the path the M2 actually runs
(Metal), collapse is free — so if the behavior is *better*, it is pure upside.

## Two SEPARATE properties (do not conflate)
1. **Transform-divergence** (collapse changes behavior vs stock): survives
   without permute. Universal, GPU-native, free.
2. **Run-to-run seasoning** (the "3 different MD5s at fixed seed" flavor from
   CLAUDE.md): does NOT come from prune+amplify — it is owned specifically by
   the entropy-seeded permute (`mftb`/`cntvct_el0`). Tier-1 is fully
   deterministic (ON reproduces byte-identically).

## METHODOLOGICAL LANDMINE (record for any future GPU collapse test)
With `-ngl 99 -fa auto`, Metal enables **flash attention**, which fuses softmax
into its own kernel and NEVER calls `kernel_soft_max`. The edited kernel is off
the execution path — a first A/B AND a destructive "zero all scores" test both
came back byte-identical for this reason. Proven by: zero-all with `-fa off`
produced garbage output, confirming the kernel executes. **Any softmax-level
collapse test on GPU must use `-fa off`, or hook the flash-attention kernel
instead.** Otherwise you get a confident false negative.

## Implication for the collapse paper (v2, DOI 10.5281/zenodo.21282030)
The paper leans on `vec_perm`'s non-bijunctive single-instruction advantage. The
SIMD efficiency is real for CPU inference, but this result shows the **behavioral
claims should be decoupled from the permutation claims**: the behavior comes from
prune+amplify (constraint-bound selection), which is universal and free on GPU,
not from the permutation. Framed honestly, this is a STRONGER result —
"constraint-bound selection is a universal, near-free attention modifier" — not a
weaker one.

## Open question (now purely about quality, cost being settled)
Is the changed behavior *better*, measured by PSE markers (NOI/DR/ACS/MCI), not
just different? That is the next experiment, and it is clean because cost is off
the table on the GPU path.
