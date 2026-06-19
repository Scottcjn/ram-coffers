# Changelog

All notable changes to **RAM Coffers** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

RAM Coffers is a NUMA-distributed conditional-memory architecture for LLM inference, originally targeting IBM POWER8 (147 t/s, 8.8× stock `llama.cpp`) and since ported to Apple Silicon (M2, 1.3× stock on Q4_K). The project is part of the [Proof of Physical AI](https://github.com/Scottcjn/Rustchain) stack.

---

## [Unreleased]

Tracking the next release on `main`. See open issues and PRs on the [GitHub repo](https://github.com/Scottcjn/ram-coffers).

---

## [0.5.0] — 2026-05-18 — *Operability & DePIN positioning*

This line of work moves the project from "research code that beats `llama.cpp`" to "operator tooling for a verified-inference DePIN node." It is the version IBM POWER10 evaluators are likely to read first.

### Added
- **NUMA topology visualization CLI** (`#651`) — `ram-coffers topology` prints a live ASCII map of detected NUMA nodes, RAM capacities, and coffer assignments. Built on top of the existing routing code; no kernel changes.
- **Reproducible NUMA-parser test suite** (`#654`, `#658`) — Topology tests now run on hosts without a real POWER8 by replaying canned `/sys/devices/system/node/` fixtures. PRs by `pagefarms` and `saim256`.
- **Treasury FAQ** (`#659`) — Documents how the RTC reward stream interacts with verified-inference workloads on the same physical node (DePIN model clarification for evaluators).
- **PPA integration guide** (`docs/PPA_INTEGRATION.md`) — How a RAM Coffers node anchors verified inference outputs into the RustChain attestation flow.

### Changed
- **DePIN repositioning** — README, NOTICE, and ecosystem links now consistently describe RAM Coffers as the *inference layer of the Proof of Physical AI stack*, not just a POWER8 optimization. The hardware that runs inference is the same hardware that mines RTC via Proof of Antiquity.
- **Documentation hygiene** (`#652`, `#653`, `#656`) — Fixed broken `elyan-prime` cross-link, corrected Apache 2.0 license wording, repaired internal anchor links.

### Fixed
- NUMA parser: handled the edge case where `/sys/devices/system/node/node0/cpulist` returns a hyphenless single-CPU string (was previously crashing on single-core VMs used for CI).

---

## [0.4.0] — 2026-03-28 — *The Matmul Collapse equation*

The scientific peak of the project so far. Three independent AI reviews (Claude, Codex, Gemini) converged on the same kernel-level equation describing what `vec_perm`-based non-bijunctive collapse is actually computing — and a stock-shape Q4_K kernel implementing it beats `llama.cpp` by **1.3× on Apple M2**. The "Architecture-General" paper (Zenodo DOI [10.5281/zenodo.19040847](https://doi.org/10.5281/zenodo.19040847)) was published the same week.

### Added
- **`EQUATION.md`** — Three-AI consensus derivation of the non-bijunctive collapse kernel. The winning form is "stock kernel + reduced K" — i.e., the optimization is *not* a different shader but a smaller inner-product dimension after path pruning.
- **Architecture-General Non-Bijunctive Hebbian Collapse paper** — Zenodo DOI [10.5281/zenodo.19040847](https://doi.org/10.5281/zenodo.19040847). Demonstrates the same idea ports from POWER8 (`vec_perm`) to Apple Silicon (ARM NEON / Metal) without algorithmic change.
- **Apple Silicon PSE port** — Non-bijunctive collapse via ARM NEON, plus a Metal sparse-attention shader for unified-memory sparse FFN. M2 setup script and `llama.cpp` integration guide included.
- **vcipher hardware AES primitive** — Alternative PSE collapse primitive using POWER8's `vcipher` instruction (timing-side-channel-resistant, separate from `vec_perm`).
- **Selective per-layer pruning** at 60% on middle layers, with quality verified on the Mac M2 benchmark suite (Qwen 2.5 7B, Qwen 3.5 4B/9B).
- **Reproducible benchmark harness** (`#56`) — `benchmark_harness.sh` reproduces the headline numbers (147 t/s POWER8, 1.3× M2) on any host with the right ggml backend.
- **GPU-native collapse handoff doc** — Exact Q4_K dequant + collapse code for porting to CUDA / Metal compute pipelines.

### Changed
- Replaced the original PSE-jitter primitive with explicit Hebbian intelligent collapse — the collapse is now an interpretable single-cycle `vec_perm` operation, not a stochastic noise injection.
- `pse-minimal.h` extracted as a drop-in header for `llama.cpp` C++ integration; Makefile updated for the new layout.

### Benchmarks (M2, Q4_K, this release)
| Model | Stock `llama.cpp` | RAM Coffers (collapsed) | Ratio |
|---|---|---|---|
| Qwen 2.5 7B | baseline | **1.3× faster** | live-measured, see `benchmarks/` |
| Qwen 3.5 4B | baseline | quality preserved at 60% prune | LPIPS / perplexity in benchmark report |

---

## [0.3.0] — 2026-03-08 — *Governance, certification, and CI*

Project graduated from solo lab notebook to public-contributor project. This is the release IBM evaluators expect for "production-readable open source."

### Added
- **BCOS certification** — Beacon Certified Open Source v1 attestation; v2 live verification added 2026-03-21 (`BCOS.md`).
- **`SECURITY.md`** — Coordinated-disclosure policy and safe-harbor language (2026-02-19, finalized 2026-03-01 as `#18`).
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant 3.0.
- **`CONTRIBUTING.md`** (`#28`) — Build instructions, NUMA test setup, PR conventions.
- **GitHub Issue Templates** (Bounty `#12` → PR `#20`) — Bug report, feature request, NUMA-specific repro template.
- **C-build CI workflow** (Bounty `#14` → PR `#21`) — Compiles and runs the topology suite on every push.
- **Architecture diagram in README** (Bounty `#13` → PR `#22`).
- **New-reader orientation path** (`#19`) — README intro tailored for evaluators who arrive cold.
- **POWER8 quick-start guide** (`#39`) — Single-page onboarding.

### Changed
- `README.md` now opens with the DePIN framing and links to the Proof of Physical AI stack.

---

## [0.2.0] — 2026-02-13 — *Papers, DOIs, and licensing*

Five papers received Zenodo DOIs; project re-licensed under Apache 2.0 with a NOTICE file establishing the priority claim that later became material when DeepSeek's "Memory Engram" paper appeared.

### Added
- **Apache 2.0 LICENSE** (2026-02-02) and **`NOTICE`** with Section 4(d) priority claim.
- **Zenodo DOIs for 5 papers** (2026-02-12):
  - RAM Coffers: NUMA-Distributed Weight Banking — DOI [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905)
  - Non-Bijunctive Permutation Collapse — DOI [10.5281/zenodo.18623920](https://doi.org/10.5281/zenodo.18623920)
  - PSE Hardware Entropy for Behavioral Divergence — DOI [10.5281/zenodo.18623922](https://doi.org/10.5281/zenodo.18623922)
  - Neuromorphic Prompt Translation (GRAIL-V) — DOI [10.5281/zenodo.18623594](https://doi.org/10.5281/zenodo.18623594)
  - RustChain: One CPU, One Vote — DOI [10.5281/zenodo.18623592](https://doi.org/10.5281/zenodo.18623592)
- **Memory Scaffolding paper** (2026-02-28) — Persistent context effects on LLM inference; DOI [10.5281/zenodo.18817988](https://doi.org/10.5281/zenodo.18817988).
- **Press / Dev.to / Grokipedia references** linked from README.

---

## [0.1.0] — 2026-01-21 — *Initial public release*

First public version. Establishes the architecture, the priority claim, and the first round of NYSE-engineer code-review feedback.

### Added
- **Initial commit** (2026-01-19) — Conditional Memory via O(1) Lookup. Four NUMA-bank coffer layout for the POWER8 S824 (Heavy/General, Science/Tech, Creative, Niche/History).
- **Neuromorphic NUMA Coffers** (2026-01-21) — Brain-hemisphere cognitive routing layer over the raw NUMA banks.
- **Dec 17, 2025 video evidence** (2026-01-21) — YouTube screenshot showing "NUMA Coffers" labeled in the dual-G4/POWER8 demo, **26 days before** DeepSeek's Engram preprint. Material for the priority claim later codified in the `NOTICE`.
- **NYSE engineering code-review optimizations** (2026-01-20) — Lock-free coffer-activate path, cache-line padding on the bank descriptors, and the dcbt prefetch hints that later became the "DCBT Resident" technique.
- **GRAIL-V paper** (2026-01-27) — Emotional prompting for efficient video generation. Same NUMA principle, applied to limbic-gated retrieval. Hopfield attractor analysis for emotional vs literal embeddings included.

---

## Priority and prior art

DeepSeek's "Memory Engram" preprint (arXiv:2601.07372, 2026-01-12) and the RAM Coffers initial commit (2026-01-19) describe convergent architectures: both separate static knowledge from dynamic computation via O(1) routed lookup. RAM Coffers' contribution is the *NUMA-explicit* form — knowledge is partitioned not just by domain but by *physical memory bank*, and routing decisions are made before the first cache-line fetch. The Dec 17, 2025 video evidence (commit `1a3b...`, file `evidence/`) predates the Engram preprint by 27 days, with the design itself dating to the POWER8 S824 deployment in November 2025.

The two lines of work are independently arrived at and we treat them as parallel rather than derivative. The `NOTICE` file documents this in the form expected by Apache 2.0 Section 4(d).

---

[Unreleased]: https://github.com/Scottcjn/ram-coffers/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Scottcjn/ram-coffers/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Scottcjn/ram-coffers/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Scottcjn/ram-coffers/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Scottcjn/ram-coffers/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Scottcjn/ram-coffers/releases/tag/v0.1.0
