# References and prior work

What this design borrows, and from whom. Grouped by what it was borrowed *for*.

## The model

- **DeepSeek-V4.** DeepSeek-AI, 2026.
  [arXiv:2606.19348](https://arxiv.org/abs/2606.19348).
  The architecture both V4 profiles implement: conv-compressed self-attention
  (CSA) interleaved with hyper-compressed attention (HCA) over shared KV
  entries, a sliding-window branch on every layer, an FP4 lightning indexer
  selecting which compressed entries a query attends to, manifold-constrained
  hyper-connections widening the residual stream, hash-gated leading MoE
  layers, and FP4 expert weights dequantised to FP8 for compute.
  `HybridAttentionConfig` is this paper's attention; the per-layer weight terms
  in `weight_params` are this repository's reading of it, checked against the
  published parameter totals rather than term by term.
- **DeepSeek-V4-Pro and DeepSeek-V4-Flash model cards.** DeepSeek-AI, 2026.
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro> and
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>.
  The `config.json` of each is the source of every field in `DEEPSEEK_V4_PRO`
  and `DEEPSEEK_V4_FLASH`, including the verbatim `compress_ratios` schedule
  that tells the planner which layers are CSA, which are HCA, and which are
  pure sliding window. Earlier revisions of this stack extrapolated V4 Pro from
  V3 and were wrong in nearly every field; the profiles are no longer marked
  `assumed`, and the tests pin them to the cards' 1.6 T / 49 B and 284 B / 13 B.
- **DeepSeek-V3 Technical Report.** DeepSeek-AI, 2024.
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
  MLA, DeepSeekMoE with fine-grained routed experts plus an always-on shared
  expert, auxiliary-loss-free load balancing, multi-token prediction, and
  native FP8 training. `model.py`'s `deepseek-v3` profile is this paper's
  configuration, and it is the calibration case: its published parameter counts
  are what the sizing arithmetic is checked against.
- **DeepSeek-V2.** DeepSeek-AI, 2024.
  [arXiv:2405.04434](https://arxiv.org/abs/2405.04434).
  Introduces multi-head latent attention. The compressed KV cache is why an 8k
  context is affordable here at all — see `MLAConfig.kv_cache_bytes_per_token`.
- **OCP Microscaling Formats (MX) Specification v1.0.** Open Compute Project,
  2023.
  <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
  E2M1 values with one E8M0 scale per 32 elements. The scale is 6.25% on top of
  the payload, which is the difference between 0.5 and 0.53125 bytes per expert
  parameter and, at 1.6 T parameters, about 46 GiB — four consoles.
- **DeepSeekMoE.** Dai et al., 2024.
  [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
  Fine-grained expert segmentation and shared-expert isolation. Small experts
  are what make an expert a *unit of placement* small enough to fit a console;
  the shared expert is why every shelf host must hold one locally.
- **FP8-LM: Training FP8 Large Language Models.** Peng et al., 2023.
  [arXiv:2310.18313](https://arxiv.org/abs/2310.18313).
- **FP8 Formats for Deep Learning.** Micikevicius et al., 2022.
  [arXiv:2209.05433](https://arxiv.org/abs/2209.05433).
  The E4M3/E5M2 definitions. `kernels/fp8.c` and `gen9_cluster/fp8.py`
  implement E4M3FN as used by DeepSeek's checkpoints and by
  `torch.float8_e4m3fn`.

## Running big MoE models on hardware that should not fit them

- **KTransformers.** KVCache-AI / Tsinghua MADSys.
  <https://github.com/kvcache-ai/ktransformers>.
  The CPU/GPU hybrid MoE playbook this stack applies across machines instead of
  within one: keep attention, router and shared expert on the fast device, keep
  routed experts on the slow large device, move activations rather than
  weights. The shelf-host/expert-member division in `planner.py` is this idea
  with a network in the middle.
- **llama.cpp** MoE CPU offload (`--n-cpu-moe`, `--override-tensor`).
  <https://github.com/ggml-org/llama.cpp>.
  Establishes that per-tensor placement across memory tiers is the practical
  knob, and that mmap plus the page cache is a serviceable weight-streaming
  tier. `ShardStore`'s mmap'd NVMe tier follows this rather than building an
  expert cache with its own eviction policy.
- **DeepSpeed-Inference / ZeRO-Inference.** Aminabadi et al., SC 2022.
  [arXiv:2207.00032](https://arxiv.org/abs/2207.00032).
  Pipeline plus expert parallelism across heterogeneous memory, and the
  observation that MoE inference is bound by weight movement rather than FLOPs.
- **FlexGen: High-Throughput Generative Inference with a Single GPU.**
  Sheng et al., ICML 2023. [arXiv:2303.06865](https://arxiv.org/abs/2303.06865).
  The GPU/CPU/disk tiering model behind the fast → slow → NVMe fallback order.
- **Petals: Collaborative Inference of Large Models.** Borzunov et al., 2022.
  [arXiv:2209.01188](https://arxiv.org/abs/2209.01188).
  Pipeline stages over commodity nodes with a real network between them, and
  the failure semantics that follow: a stateless stage can be re-routed, a
  stage holding cache state cannot. `dispatch.py` and `coordinator.py` take the
  same position for expert consoles versus shelf hosts.
- **Mixtral of Experts.** Jiang et al., 2024.
  [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
  Expert-locality measurements motivating throughput-weighted placement.
- **Fiddler**, Kamahori et al., 2024.
  [arXiv:2402.07033](https://arxiv.org/abs/2402.07033), and
  **MoE-Infinity**, Xue et al., 2024.
  [arXiv:2401.14361](https://arxiv.org/abs/2401.14361).
  Two treatments of expert activation skew — the reason NVMe-resident experts
  are workable rather than uniformly fatal, since the page cache retains the
  ones that are actually routed to.

## RAM Coffers itself

- **RAM Coffers: NUMA-Distributed Weight Banking.** Boudreaux, 2026.
  [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905).
  The parent architecture. This stack applies the coffer idea one level up: a
  coffer becomes a console memory tier, and routing happens across consoles
  rather than NUMA banks. See the repository root README.
- The `kimi-k3-playstation3` branch of this repository, for the P3XC protocol
  that G9XC descends from and for the shelf/coordinator structure. The
  adaptations are documented in [G9XC.md](G9XC.md).

## Console hardware

Console silicon specifications are vendor-disclosed rather than measured here;
figures in `hardware.py` are marked `UNVERIFIED` where that is the case.

- **AMD RDNA 2 Instruction Set Architecture Reference Guide**, AMD, 2020, and
  the RDNA 2 whitepaper — CU structure and clock behaviour for the GPUs in this
  family.
- **Zen 2 microarchitecture.** Suggs, Subramony & Bouvier, *IEEE Micro* 40(2),
  2020. [doi:10.1109/MM.2020.2974217](https://doi.org/10.1109/MM.2020.2974217).
  AVX2 without AVX-512, which is why `kernels/expert_avx2.c` targets FMA3/AVX2
  and nothing wider.
- **Hot Chips 32 (2020)**: the Xbox Series X SoC ("Scarlett") and PS5 SoC
  presentations — CU counts, the Series X split 10 GB/6 GB memory topology, and
  the PS5's variable-clock scheme.
- Sony, *The Road to PS5* (2020) — 448 GB/s unified GDDR6, the 5.5 GB/s NVMe
  and its decompression path.
- **ps5-linux-loader** and the PS5 amdgpu (GFX1013) work.
  <https://github.com/ps5-payload-dev>. Establishes what a PS5 node can
  actually run: FW 3.00–7.61, phat and slim, amdgpu with GFX1013.
- **BC-250**: the Sony Oberon-derived blade, and the community Linux/amdgpu
  work on it. The member of this family whose GPU compute path is most
  routinely exercised, hence its role in the fleet as the sanity check for the
  Vulkan path.
- AMD 4700S / 4800S desktop kits: console SoCs with the GPU fused off, GDDR6 as
  system memory at greatly reduced effective bandwidth. Different harvests, not
  revisions of each other — the 4700S is a PS5 "Ariel" die
  ([Tom's Hardware teardown](https://www.tomshardware.com/reviews/amd-4700s-desktop-kit-review-ps5-cpu),
  which also measures the GDDR6 at 92.9 GB/s copy and 145 ns), the 4800S a
  Series X one
  ([Digital Foundry](https://www.digitalfoundry.net/articles/digitalfoundry-2023-amd-4800s-desktop-kit-review-play-pc-games-on-the-xbox-series-x-cpu)).
  Modelled as CPU-only nodes with a large slow coffer.

  A card in the slot does not make either one a small PS5. Both slots are x4:
  PCIe 2.0 x4 (~2 GB/s) on the 4700S, PCIe 4.0 x4 (~7.9 GB/s) on the 4800S. A
  GPU there can only run experts that fit in its own VRAM, because reaching the
  board's 15 GB across that link is two orders of magnitude slower than the
  console memory the design assumes. Supporting it properly needs a memory model
  where a tier's bandwidth depends on which backend is reading it — the same
  change the Steam Machine's split GDDR6/DDR5 would need, and not yet made.

## Compute backends

- **ROCm** on gfx1013. Debian's ROCm packaging does run on this target, but
  library coverage is uneven: some libraries do not ship the target and others
  fall back to a lower capability level because their build scripts assume a
  higher gfx tier. This is why `ComputeBackend.ROCM.throughput_is_assumed` is
  true and a ROCm node must supply `measured_gemv_gflops`.
  `kernels/expert_hip.hip` avoids rocBLAS, Tensile and hipBLASLt for exactly
  this reason.
- **Mesa RADV** Vulkan compute — the default GPU path. No tuned-library
  assumptions, and it is the driver the PS5 Linux work actually uses.
- **Vulkan 1.1** compute shaders, `VK_KHR_8bit_storage` for the FP8 code path.

## A note on attribution

Where this stack does something the same way as one of the above, it is because
that work established it, not because it was arrived at independently. The
places where it diverges — batching a console's whole expert set into one
frame, refusing to fail over a shelf host, requiring measured throughput for
ROCm nodes — are diverging *from* these designs for reasons given in the code
comments at each site.
