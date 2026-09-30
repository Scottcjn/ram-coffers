"""DeepSeek model profiles: what the planner needs to know about a checkpoint.

A DeepSeek decoder splits across consoles according to four properties, and they
pull in different directions:

1. **Compressed attention.** V3 uses MLA, which squeezes the KV cache into one
   ``kv_lora_rank``-wide latent plus a decoupled RoPE key per token. V4 replaces
   it with a *hybrid* of CSA and HCA (:class:`HybridAttentionConfig`), which
   compresses along the sequence instead: every ``m`` tokens become one cache
   entry. Either way the cache is small enough that a 16 GB console can host
   attention, and either way the block's own weights are read for every token,
   so they belong in a fast coffer.
2. **DeepSeekMoE** puts many narrow routed experts plus one always-on shared
   expert in each MoE layer. Only ``top_k`` of the routed experts run per token,
   so the routed weights are enormous but cold — exactly what a bandwidth-tiered
   coffer hierarchy and an NVMe are for.
3. **Mixed-precision weights.** V3 is FP8 (E4M3) throughout. V4 keeps FP8 for
   everything read every token and drops the *expert* weights to MXFP4, which is
   where nearly all the bytes are: it roughly halves what a fleet has to hold.
   :class:`QuantSpec` carries the format and its block scales, so the packed
   size — the thing the planner actually reasons about — follows from the
   checkpoint's own quantisation config rather than from a guess.
4. **Residual width.** V4's manifold-constrained hyper-connections widen the
   residual stream to ``hc_mult`` × hidden while the layer input stays hidden-
   wide. The layer does not get more expensive, but a *pipeline boundary* does:
   what crosses between shelves is the whole residual state, so a shelf hop
   carries ``hc_mult`` times the activation it would in V3.

Sizes here are computed from the architecture, not read off a file listing, so a
profile can be edited (or a new one declared) and every derived number follows.
Each profile's :meth:`ModelProfile.total_params` is checked against the published
parameter count in the tests, which is what makes it safe to trust the derived
byte counts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

#: Bytes per parameter for the formats a console can hold, *before* block
#: scales; :class:`QuantSpec` adds those.
DTYPE_BYTES = {"fp8": 1.0, "fp4": 0.5, "mxfp4": 0.5,
               "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}


@dataclass(frozen=True)
class QuantSpec:
    """A weight format together with the block scales stored alongside it.

    Quantised weights are never only their payload: a blockwise format keeps one
    scale per block, and whether that rounds to nothing or to 6% depends
    entirely on the block shape. FP8 with an fp32 scale per 128x128 tile is
    0.02% overhead; MXFP4 with a one-byte exponent per 32 values is 6.25%, which
    is the difference between an expert fitting a coffer and not.
    """

    dtype: str
    #: Bytes per stored scale (4 for fp32, 1 for the ue8m0 / E8M0 exponents).
    scale_bytes: float = 0.0
    #: Elements sharing one scale; 0 means the format carries no scales.
    scale_block: int = 0

    @property
    def bytes_per_param(self) -> float:
        base = DTYPE_BYTES[self.dtype]
        if self.scale_block:
            base += self.scale_bytes / self.scale_block
        return base


#: FP8 E4M3 with an fp32 scale per 128x128 tile, as V3/R1 ship it.
FP8_BLOCK128 = QuantSpec("fp8", scale_bytes=4.0, scale_block=128 * 128)
#: FP8 E4M3 with a ue8m0 scale per 128x128 tile, as V4 ships it.
FP8_UE8M0 = QuantSpec("fp8", scale_bytes=1.0, scale_block=128 * 128)
#: FP8 E4M3 with a ue8m0 scale per 32x32 tile, as V4.1 ships it. Finer tiles
#: are ~0.1% of scales rather than ~0.006% — still nothing next to the FP4
#: experts.
FP8_UE8M0_32 = QuantSpec("fp8", scale_bytes=1.0, scale_block=32 * 32)
#: OCP MXFP4: E2M1 values with one E8M0 scale per 32-element tile.
MXFP4 = QuantSpec("mxfp4", scale_bytes=1.0, scale_block=32)
#: V4.1's KV-cache format: E2M1 values with one E4M3 scale per 16 channels.
#: Not the OCP tile shape — 0.5625 bytes per cached element.
FP4_E4M3_16 = QuantSpec("fp4", scale_bytes=1.0, scale_block=16)


class AttentionConfig:
    """What the planner needs from an attention block, per layer.

    Subclasses differ in how much cache a token leaves behind and how much of
    that cache the *next* token has to read, which are not the same number once
    attention is sparse. Everything is per layer, because V4 interleaves layer
    kinds that differ by a factor of 32 in cache size.
    """

    n_heads: int

    def kind(self, layer: int = 0) -> str:
        """Short name of this layer's attention, for humans and for tests."""
        raise NotImplementedError

    def weight_params(self, hidden_size: int, layer: int = 0) -> int:
        """Parameters in one attention block."""
        raise NotImplementedError

    def kv_cache_bytes_per_token(self, dtype: str = "bf16",
                                 layer: int = 0) -> int:
        """Cache one more token *adds* to this layer. May be zero."""
        raise NotImplementedError

    def state_bytes(self, dtype: str = "bf16", layer: int = 0) -> int:
        """Cache this layer holds regardless of context length."""
        return 0

    def kv_read_bytes(self, context_tokens: int, dtype: str = "bf16",
                      layer: int = 0) -> int:
        """Cache one decoded token has to *read* — not what it stores."""
        return (context_tokens * self.kv_cache_bytes_per_token(dtype, layer)
                + self.state_bytes(dtype, layer))


@dataclass(frozen=True)
class MLAConfig(AttentionConfig):
    """Multi-head latent attention (V2/V3) shape."""

    n_heads: int
    kv_lora_rank: int
    q_lora_rank: Optional[int]
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def kind(self, layer: int = 0) -> str:
        return "mla"

    def kv_cache_bytes_per_token(self, dtype: str = "fp8",
                                 layer: int = 0) -> int:
        """One layer's KV cache cost for one token.

        MLA stores the compressed latent (``kv_lora_rank``) plus the decoupled
        RoPE key (``qk_rope_head_dim``), shared across heads — the whole point of
        the architecture, and the reason a console can hold a long context.
        """
        return int(round(self.kv_lora_rank * DTYPE_BYTES[dtype]
                         + self.qk_rope_head_dim * DTYPE_BYTES["bf16"]))

    def weight_params(self, hidden_size: int, layer: int = 0) -> int:
        """Parameters in one MLA block."""
        q_out = self.n_heads * self.qk_head_dim
        if self.q_lora_rank is None:
            q = hidden_size * q_out
        else:
            q = (hidden_size * self.q_lora_rank
                 + self.q_lora_rank * q_out)
        # Joint KV down-projection (latent + decoupled RoPE key), then up.
        kv_down = hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_up = (self.kv_lora_rank * self.n_heads
                 * (self.qk_nope_head_dim + self.v_head_dim))
        out = self.n_heads * self.v_head_dim * hidden_size
        return q + kv_down + kv_up + out


@dataclass(frozen=True)
class HybridAttentionConfig(AttentionConfig):
    """V4's interleaved CSA / HCA attention, with a sliding-window branch.

    Both halves compress the cache *along the sequence*: CSA folds every ``m``
    tokens into one shared-KV entry and then attends sparsely to ``index_topk``
    of those entries, chosen by a small FP4 indexer; HCA folds every ``m'``
    (≫ ``m``) tokens into one entry and attends to all of them densely. A layer
    is one or the other, given by :attr:`compress_ratios`; a ratio of 0 means the
    layer has no compressed branch at all and runs pure sliding-window
    attention.

    The consequence the planner cares about is that *stored* and *read* cache
    diverge. An HCA layer at a million tokens stores 16 KiB and reads all of it;
    a CSA layer stores 512 KiB but reads only the selected entries plus one
    indexer scan. Modelling them with a single number gets long-context decode
    wrong by more than an order of magnitude.

    ``compress_ratios`` carries one entry per *block*, which is one more than
    ``n_layers`` when the checkpoint has an MTP head; the trailing entry is that
    head's block.
    """

    n_heads: int
    #: ``c``: width of a shared KV entry, and of each query head.
    head_dim: int
    #: ``d_c``: queries are produced through this low-rank bottleneck.
    q_lora_rank: int
    #: ``g`` and ``d_g``: the output projection is grouped, not one big matrix.
    o_groups: int
    o_group_dim: int
    #: How many of ``head_dim``'s dimensions are rotated. Partial RoPE, so this
    #: is a subset of the head, not extra width.
    qk_rope_head_dim: int
    #: CSA's indexer: query heads, head dim, and how many entries it selects.
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    #: Sliding-window branch present on every layer.
    sliding_window: int
    #: ``m`` and ``m'``.
    csa_ratio: int
    hca_ratio: int
    #: Per-block compression ratio: ``csa_ratio``, ``hca_ratio``, or 0 for a
    #: layer that is pure sliding window.
    compress_ratios: Tuple[int, ...] = ()
    #: The indexer's keys are cached and multiplied in FP4.
    index_quant: QuantSpec = MXFP4

    def ratio(self, layer: int = 0) -> int:
        if not self.compress_ratios:
            return self.csa_ratio
        index = min(layer, len(self.compress_ratios) - 1)
        return self.compress_ratios[index]

    def kind(self, layer: int = 0) -> str:
        ratio = self.ratio(layer)
        if ratio <= 1:
            return "swa"
        return "csa" if ratio == self.csa_ratio else "hca"

    def weight_params(self, hidden_size: int, layer: int = 0) -> int:
        d, c, nh = hidden_size, self.head_dim, self.n_heads
        ratio = self.ratio(layer)
        # Queries: down to d_c, then up to one c-wide query per head.
        q = d * self.q_lora_rank + self.q_lora_rank * nh * c
        # Base KV projection: the sliding-window branch exists on every layer,
        # including ones that also carry a compressed branch.
        kv_base = d * c
        # Grouped output: heads within a group share an intermediate of width
        # d_g, and each group projects back to d.
        out = nh * c * self.o_group_dim + self.o_groups * self.o_group_dim * d
        params = q + kv_base + out + self.q_lora_rank + c + nh
        if ratio <= 1:
            # Pure sliding-window attention: one KV entry, no compression.
            return params
        # Heavily or sparsely compressed KV: two gating projections plus a
        # per-compression-window positional bias.  CSA uses overlapping windows.
        coff = 2 if ratio == self.csa_ratio else 1
        kv_compress = coff * d * c + coff * d * c + ratio * coff * c + c
        params += kv_compress
        if ratio == self.csa_ratio:
            # Lightning indexer: queries from the q-lora bottleneck, per-head
            # weights from the hidden state, and its own overlapping compressor.
            idx_c = self.index_head_dim
            idx_coff = 2
            params += (self.q_lora_rank * self.index_n_heads * idx_c
                       + d * self.index_n_heads
                       + idx_coff * d * idx_c
                       + idx_coff * d * idx_c
                       + ratio * idx_coff * idx_c
                       + idx_c)
        return params

    def kv_entry_bytes(self, dtype: str = "fp8") -> float:
        """Bytes of one cached KV entry: FP8 for the non-RoPE part, BF16 for
        the RoPE part, matching the V4 checkpoint format."""
        nope = self.head_dim - self.qk_rope_head_dim
        return nope * DTYPE_BYTES[dtype] + self.qk_rope_head_dim * DTYPE_BYTES["bf16"]

    def kv_cache_bytes_per_token(self, dtype: str = "fp8",
                                 layer: int = 0) -> int:
        kind = self.kind(layer)
        if kind == "swa":
            return 0  # the window is a fixed-size state, see state_bytes
        ratio = self.ratio(layer)
        entry = self.kv_entry_bytes(dtype) / ratio
        if kind == "csa":
            entry += (self.index_head_dim
                      * self.index_quant.bytes_per_param / ratio)
        return int(round(entry))

    def state_bytes(self, dtype: str = "fp8", layer: int = 0) -> int:
        """The sliding-window branch, which every layer carries.

        Uncompressed tail tokens live here too; both are bounded, which is why
        DeepSeek's own serving stack treats them as a fixed pool rather than as
        part of the growing cache.
        """
        return int(round(self.sliding_window * self.kv_entry_bytes(dtype)))

    def kv_read_bytes(self, context_tokens: int, dtype: str = "fp8",
                      layer: int = 0) -> int:
        kind = self.kind(layer)
        state = self.state_bytes(dtype, layer)
        if kind == "swa":
            return state
        ratio = self.ratio(layer)
        entries = context_tokens / ratio
        if kind == "hca":
            return int(round(entries * self.kv_entry_bytes(dtype) + state))
        # CSA scans every compressed entry with the FP4 indexer, then reads only
        # the top-k it selected.
        scan = (entries * self.index_head_dim
                * self.index_quant.bytes_per_param)
        selected = min(entries, float(self.index_topk))
        return int(round(scan + selected * self.kv_entry_bytes(dtype) + state))


@dataclass(frozen=True)
class CSA2Config(AttentionConfig):
    """V4.1's CSA2 attention: static Full / Reindex / Reuse layers over shared
    KV pools, inside a causal encoder-decoder stack.

    V4 gave every layer its own compressed stream. CSA2 instead keeps a small
    number of shared pools: a layer listed in ``kv_source_layers`` appends
    main-KV entries to its stream, a layer in ``index_source_layers`` appends
    indexer keys and computes its own top-k selection over the shared pool,
    and every other compressed layer reuses the most recent selection and
    stores nothing at all. The decoder's pool is *global*: its entries are
    projected from the encoder's final hidden states rather than grown per
    layer, which is why a decoder layer's ``compress_ratios`` entry reads 1 —
    it never writes a stream of its own.

    Two properties matter to the planner:

    - The whole model's growing KV cache lives on the few shelf hosts that own
      a source layer, not spread one stream per layer — at a million tokens
      the derived figure is ~864 B/token against the card's ~890 (a ~3%
      under-read this planner accepts; see the profile comment).
    - Later decoder indexers are confined to the candidate pool built by
      ``candidate_source_layer`` — ``candidate_topk_blocks`` x
      ``candidate_block_size`` entries — so their scan cost is bounded
      independently of context length, unlike an encoder-side scan.

    ``compress_ratios`` is verbatim from the checkpoint, one entry per block
    including the draft blocks: 0 marks a pure sliding-window block.
    """

    n_heads: int
    #: ``c``: width of a shared KV entry, and of each query head.
    head_dim: int
    q_lora_rank: int
    #: The output projection's per-group low-rank bottleneck.
    o_lora_rank: int
    o_groups: int
    #: Partial RoPE: a subset of ``head_dim`` is rotated.
    qk_rope_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    sliding_window: int
    #: ``m``: every shared stream advances one entry per this many tokens.
    csa_ratio: int
    #: Per-block schedule, verbatim from ``compress_ratios``: ``csa_ratio`` for
    #: encoder compressed layers, 1 for decoder layers (they write no stream),
    #: 0 for pure sliding-window blocks including the DSpark draft blocks.
    compress_ratios: Tuple[int, ...] = ()
    #: Blocks that append main-KV entries to the shared pool (Full, when also
    #: in ``index_source_layers``).
    kv_source_layers: Tuple[int, ...] = ()
    #: Blocks that append indexer keys and run their own selection.
    index_source_layers: Tuple[int, ...] = ()
    #: The first decoder layer, whose Full-mode pass builds the candidate pool
    #: that later decoder indexers are restricted to. A Reindex layer below it
    #: has no pool yet and scans its own stream like a Full layer does.
    candidate_source_layer: int = 0
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    #: Encoder/decoder boundary of the CED stack; blocks below this are the
    #: causal encoder.
    n_encoder_layers: int = 0
    #: V4.1 caches the shared pools in FP4, not FP8.
    kv_quant: QuantSpec = FP4_E4M3_16
    index_quant: QuantSpec = FP4_E4M3_16

    def ratio(self, layer: int = 0) -> int:
        if not self.compress_ratios:
            return self.csa_ratio
        index = min(layer, len(self.compress_ratios) - 1)
        return self.compress_ratios[index]

    def kind(self, layer: int = 0) -> str:
        if self.ratio(layer) == 0:
            return "swa"
        if layer in self.kv_source_layers:
            return "full" if layer in self.index_source_layers else "kv-src"
        if layer in self.index_source_layers:
            return "reindex"
        return "reuse"

    def weight_params(self, hidden_size: int, layer: int = 0) -> int:
        d, c, nh = hidden_size, self.head_dim, self.n_heads
        # Queries: down to q_lora_rank, then up to one c-wide query per head.
        q = d * self.q_lora_rank + self.q_lora_rank * nh * c
        # The sliding-window branch exists on every block, including draft
        # blocks and layers that otherwise share a pool.
        kv_base = d * c
        # Grouped low-rank output: heads collapse to o_lora_rank per group and
        # each group projects back to d.
        out = nh * c * self.o_lora_rank + self.o_groups * self.o_lora_rank * d
        params = q + kv_base + out + self.q_lora_rank + c + nh
        kind = self.kind(layer)
        if kind in ("swa", "reuse"):
            # Reuse layers produce no stream of their own: queries, the window
            # branch, and the output projection are the whole block.
            return params
        # An indexer: queries off the q bottleneck, per-head weights off the
        # hidden state, and its own compressor terms (same shape as V4's).
        idx_c = self.index_head_dim
        params += (self.q_lora_rank * self.index_n_heads * idx_c
                   + d * self.index_n_heads
                   + 2 * d * idx_c + 2 * d * idx_c
                   + self.csa_ratio * 2 * idx_c + idx_c)
        if kind in ("full", "kv-src"):
            # A main-KV producer adds the compression projections that fold
            # every csa_ratio tokens into one pooled entry.
            params += 2 * d * c + 2 * d * c + self.csa_ratio * 2 * c + c
        return params

    def kv_entry_bytes(self) -> float:
        """Bytes of one pooled KV entry: FP4 end to end, RoPE dims included."""
        return self.head_dim * self.kv_quant.bytes_per_param

    def index_entry_bytes(self) -> float:
        return self.index_head_dim * self.index_quant.bytes_per_param

    def kv_cache_bytes_per_token(self, dtype: str = "fp4",
                                 layer: int = 0) -> int:
        """What one more token appends to this layer's streams — zero for the
        many layers that only read a shared pool."""
        stored = 0.0
        if layer in self.kv_source_layers:
            stored += self.kv_entry_bytes() / self.csa_ratio
        if layer in self.index_source_layers:
            stored += self.index_entry_bytes() / self.csa_ratio
        return int(round(stored))

    def state_bytes(self, dtype: str = "fp4", layer: int = 0) -> int:
        """The sliding-window branch, which every block carries.

        V4.1's bounded-replay rule rebuilds this window instead of persisting
        it, but while a sequence is live the window still occupies the shelf
        host — sizing is unchanged, eviction is what changed.
        """
        return int(round(self.sliding_window * self.kv_entry_bytes()))

    def _index_scan_bytes(self, context_tokens: int, layer: int) -> float:
        """What this layer's own selection costs, where it has one.

        A Full layer scans its whole stream. A Reindex layer in the decoder
        scans only the candidate pool — the hierarchical indexer's point is
        that this stops growing with context — and an encoder Reindex layer,
        had one existed in this schedule, scans its stream like a Full layer.
        """
        entries = context_tokens / self.csa_ratio
        if (self.kind(layer) == "reindex"
                and layer >= self.candidate_source_layer
                and self.candidate_topk_blocks):
            entries = min(entries,
                          float(self.candidate_topk_blocks
                                * self.candidate_block_size))
        return entries * self.index_entry_bytes()

    def kv_read_bytes(self, context_tokens: int, dtype: str = "fp4",
                      layer: int = 0) -> int:
        """Cache one decoded token reads — selected entries, plus this layer's
        own index scan if it performs one."""
        state = self.state_bytes(dtype, layer)
        if self.kind(layer) == "swa":
            return state
        entries = context_tokens / self.csa_ratio
        selected = min(entries, float(self.index_topk)) * self.kv_entry_bytes()
        scan = 0.0
        if self.kind(layer) in ("full", "reindex"):
            scan = self._index_scan_bytes(context_tokens, layer)
        return int(round(selected + scan + state))


@dataclass(frozen=True)
class EngramConfig:
    """Conditional-memory n-gram tables (deepseek-ai/Engram, V4.1 variant).

    A table is a hash-addressed store of ``head_dim``-wide rows: each token
    looks up a handful of rows by n-gram and the result is fused into the
    hidden state at ``layer_ids``. Deterministic addressing is the property
    the planner cares about — the rows a token will touch are known *before*
    the layer runs, so the table can live on the slowest tier that still
    answers a lookup, which on this hardware is shelf-local NVMe. The tables
    are the coldest weights the model owns: ~196 B parameters read a few
    kilobytes at a time.
    """

    #: Blocks whose hidden state takes an Engram fusion.
    layer_ids: Tuple[int, ...]
    #: Rows per table, one entry per ``layer_ids`` element.
    num_embeddings: Tuple[int, ...]
    n_heads: int
    head_dim: int
    #: The n-gram hash space the rows are addressed from.
    vocab_size: int
    compressed_vocab_size: int
    max_ngram_size: int

    def table_params(self, table: int) -> int:
        return self.num_embeddings[table] * self.head_dim

    def params(self, hidden_size: int) -> int:
        """Rows, plus a per-table fusion projection back into hidden."""
        tables = sum(self.table_params(i)
                     for i in range(len(self.num_embeddings)))
        fuse = len(self.num_embeddings) * hidden_size * self.n_heads \
            * self.head_dim
        return tables + fuse

    def total_bytes(self, hidden_size: int, quant: QuantSpec) -> int:
        return int(round(self.params(hidden_size) * quant.bytes_per_param))

    def table_row_bytes(self, table: int, quant: QuantSpec) -> int:
        """The hash-addressed row store: the NVMe-resident part of a table."""
        return int(round(self.table_params(table) * quant.bytes_per_param))

    def row_unit_bytes(self, quant: QuantSpec) -> float:
        """Bytes in one row — the unit a RAM shard map is cut in."""
        return self.head_dim * quant.bytes_per_param

    def fusion_bytes(self, hidden_size: int, quant: QuantSpec) -> int:
        """The per-table projection back into hidden — read every token, so
        it belongs in the stage host's RAM, not on the drive."""
        return int(round(hidden_size * self.n_heads * self.head_dim
                         * quant.bytes_per_param))

    def table_bytes(self, table: int, hidden_size: int,
                    quant: QuantSpec) -> int:
        return (self.table_row_bytes(table, quant)
                + self.fusion_bytes(hidden_size, quant))

    def read_bytes_per_token(self, quant: QuantSpec) -> int:
        """One token's lookup: at most one row per head per n-gram size —
        kilobytes, which is why the tables can sit on NVMe at all."""
        rows = self.n_heads * self.max_ngram_size
        return int(round(rows * self.head_dim * quant.bytes_per_param))


@dataclass(frozen=True)
class VisionConfig:
    """The vision tower of a multimodal checkpoint: a ViT plus projector.

    Small, dense, and cold in the planning sense that matters — it runs once
    per image, not once per token — so it is placed as a single piece like the
    embedding tables rather than split like a decoder layer.
    """

    n_layers: int
    hidden_size: int
    n_heads: int
    intermediate_size: int
    patch_size: int
    #: Pixel-unshuffle factor before the projector (3 means 3x3 patches merge).
    downsample_ratio: int

    def params(self, out_hidden_size: int) -> int:
        h = self.hidden_size
        per_layer = 4 * h * h + 2 * h * self.intermediate_size + 2 * h
        patch_embed = 3 * self.patch_size * self.patch_size * h
        # Two-layer projector: merged patches to model width, then model width
        # to model width.
        merged = h * self.downsample_ratio * self.downsample_ratio
        projector = merged * out_hidden_size + out_hidden_size * out_hidden_size
        return self.n_layers * per_layer + patch_embed + projector

    def total_bytes(self, out_hidden_size: int, quant: QuantSpec) -> int:
        return int(round(self.params(out_hidden_size)
                         * quant.bytes_per_param))


@dataclass(frozen=True)
class MoEConfig:
    """DeepSeekMoE shape: many narrow routed experts plus shared experts."""

    n_routed_experts: int
    n_shared_experts: int
    top_k: int
    moe_intermediate_size: int
    #: Leading layers that use a dense MLP instead of MoE.
    n_dense_layers: int
    dense_intermediate_size: int
    #: Experts are grouped for device-limited routing; a token's top-k is drawn
    #: from at most ``topk_group`` of ``n_group`` groups. The planner uses this
    #: to bound how many consoles a single token can touch.
    n_group: int = 1
    topk_group: int = 1
    #: Leading MoE layers whose gate is a hash of the token rather than a
    #: learned router. Same weights and same cost, but the expert choice is
    #: known before the layer runs, so those layers can be prefetched exactly
    #: instead of speculatively.
    n_hash_layers: int = 0

    def expert_params(self) -> int:
        """One routed expert: gate, up, down."""
        return 3 * self.moe_intermediate_size

    def dense_mlp_params(self, hidden_size: int) -> int:
        return 3 * hidden_size * self.dense_intermediate_size


@dataclass(frozen=True)
class ModelProfile:
    """Everything the planner needs about a DeepSeek checkpoint."""

    name: str
    n_layers: int
    hidden_size: int
    vocab_size: int
    attention: AttentionConfig
    moe: MoEConfig
    #: Format of everything read for every token.
    weights: QuantSpec = FP8_BLOCK128
    #: Format of the routed and shared expert weights, when it differs.
    expert_weights: Optional[QuantSpec] = None
    #: Multi-token-prediction heads (V3 ships one; V4.1 ships three DSpark
    #: draft blocks). Each is a whole extra decoder block plus a head, so it
    #: is placed like a layer, not folded in.
    n_mtp_heads: int = 1
    #: The draft blocks' MoE when it differs from the backbone's — V4.1's
    #: DSpark blocks route to a narrower expert set.
    draft_moe: Optional[MoEConfig] = None
    #: DSpark's Markov-rank projection width per draft block; 0 for plain MTP.
    draft_markov_rank: int = 0
    #: Conditional-memory tables, when the checkpoint has them (V4.1).
    engram: Optional[EngramConfig] = None
    #: The vision tower, when the checkpoint is multimodal (V4.1).
    vision: Optional[VisionConfig] = None
    #: Hyper-connection expansion: the residual stream is this many times
    #: hidden-wide, which is what crosses a pipeline boundary.
    hc_mult: int = 1
    #: True when the configuration is extrapolated rather than published.
    assumed: bool = False
    assumptions: tuple = ()
    #: Where the configuration came from, printed next to the sizes.
    source: str = ""

    # -- formats -----------------------------------------------------------
    @property
    def dtype(self) -> str:
        return self.weights.dtype

    @property
    def expert_quant(self) -> QuantSpec:
        return self.expert_weights or self.weights

    @property
    def bytes_per_param(self) -> float:
        return self.weights.bytes_per_param

    @property
    def mixed_precision(self) -> bool:
        return self.expert_quant.dtype != self.weights.dtype

    # -- per-piece sizes ---------------------------------------------------
    @property
    def n_moe_layers(self) -> int:
        return max(0, self.n_layers - self.moe.n_dense_layers)

    @property
    def planning_layers(self) -> int:
        """Blocks the planner has to place.

        An MTP head is a whole decoder block plus a projection, not a small
        appendage, so it is placed exactly like a layer rather than treated as
        one indivisible lump that no console can hold.
        """
        return self.n_layers + self.n_mtp_heads

    def moe_for_block(self, index: int) -> MoEConfig:
        """The MoE governing block ``index``: the draft MoE for blocks past
        ``n_layers`` when one is configured, else the backbone's."""
        if index >= self.n_layers and self.draft_moe is not None:
            return self.draft_moe
        return self.moe

    def expert_bytes(self, index: Optional[int] = None) -> int:
        """One routed expert, packed, block scales included."""
        moe = self.moe if index is None else self.moe_for_block(index)
        params = moe.expert_params() * self.hidden_size
        return int(round(params * self.expert_quant.bytes_per_param))

    def shared_expert_bytes(self, index: Optional[int] = None) -> int:
        moe = self.moe if index is None else self.moe_for_block(index)
        return self.expert_bytes(index) * moe.n_shared_experts

    def attention_bytes(self, layer: int = 0) -> int:
        params = self.attention.weight_params(self.hidden_size, layer)
        return int(round(params * self.bytes_per_param))

    def _hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    def _mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def _hc_params(self) -> int:
        """Float parameters in one block's two hyper-connection mappings."""
        return 2 * (self._mix_hc() * self._hc_dim() + self._mix_hc() + 3)

    def _hc_bytes(self) -> int:
        return int(round(self._hc_params() * self.bytes_per_param))

    def _final_head_params(self) -> int:
        params = self.hidden_size
        if self.hc_mult > 1:
            params += self.hc_mult * self._hc_dim() + self.hc_mult + 1
        return params

    def _final_head_bytes(self) -> int:
        hc_params = 0
        if self.hc_mult > 1:
            hc_params = self.hc_mult * self._hc_dim() + self.hc_mult + 1
        return int(round(self.hidden_size * DTYPE_BYTES["bf16"]
                         + hc_params * self.bytes_per_param))

    def _mtp_extra_params(self) -> int:
        params = 3 * self.hidden_size
        if self.hc_mult > 1:
            params += self.hc_mult * self._hc_dim() + self.hc_mult + 1
        return params

    def _mtp_extra_bytes(self) -> int:
        hc_params = 0
        if self.hc_mult > 1:
            hc_params = self.hc_mult * self._hc_dim() + self.hc_mult + 1
        return int(round(3 * self.hidden_size * DTYPE_BYTES["bf16"]
                         + hc_params * self.bytes_per_param))

    def router_bytes(self, index: Optional[int] = None) -> int:
        """Router matrix; the bias/hash table is added per-layer."""
        moe = self.moe if index is None else self.moe_for_block(index)
        return int(round(self.hidden_size * moe.n_routed_experts
                         * self.bytes_per_param))

    def dense_mlp_bytes(self) -> int:
        raw = self.moe.dense_mlp_params(self.hidden_size) * self.bytes_per_param
        return int(round(raw))

    def hot_bytes_per_moe_layer(self, layer: int = 0) -> int:
        """Weights an MoE layer reads for *every* token.

        Attention + router + shared expert + input/post norms +
        hyper-connections + the gate's hash table or bias. This is the quantity
        that must land in a fast coffer; the routed experts are the cold
        remainder.
        """
        norms = int(round(2 * self.hidden_size * DTYPE_BYTES["bf16"]))
        moe = self.moe_for_block(layer)
        gate_extras = 0
        if layer >= moe.n_dense_layers:
            if layer < moe.n_hash_layers:
                gate_extras = int(round(self.vocab_size * moe.top_k * 4))
            else:
                gate_extras = int(round(moe.n_routed_experts
                                        * self.bytes_per_param))
        hc = self._hc_bytes() if self.hc_mult > 1 else 0
        return (self.attention_bytes(layer) + self.router_bytes(layer)
                + gate_extras + self.shared_expert_bytes(layer)
                + norms + hc)

    def cold_bytes_per_moe_layer(self, index: Optional[int] = None) -> int:
        moe = self.moe if index is None else self.moe_for_block(index)
        return self.expert_bytes(index) * moe.n_routed_experts

    def dense_layer_bytes(self, layer: int = 0) -> int:
        norms = int(round(2 * self.hidden_size * DTYPE_BYTES["bf16"]))
        hc = self._hc_bytes() if self.hc_mult > 1 else 0
        return self.attention_bytes(layer) + self.dense_mlp_bytes() + norms + hc

    def embedding_bytes(self) -> int:
        return int(round(self.vocab_size * self.hidden_size
                         * DTYPE_BYTES["bf16"]))

    def lm_head_bytes(self) -> int:
        return self.embedding_bytes()

    def mtp_projection_bytes(self) -> int:
        """The extra projection an MTP head carries on top of a decoder block."""
        return int(round(2 * self.hidden_size * self.hidden_size
                         * self.bytes_per_param))

    def mtp_hot_bytes(self, head: int = 0) -> int:
        """Per-token weights of one draft block: its hot block, projection,
        and — for DSpark — the Markov-rank projection."""
        dspark = int(round(self.draft_markov_rank * self.hidden_size
                           * self.bytes_per_param))
        return (self.hot_bytes_per_moe_layer(self.n_layers + head)
                + self.mtp_projection_bytes() + self._mtp_extra_bytes()
                + dspark)

    def mtp_bytes(self) -> int:
        """All draft blocks, weights in full."""
        if self.n_mtp_heads == 0:
            return 0
        return sum(self.mtp_hot_bytes(head)
                   + self.cold_bytes_per_moe_layer(self.n_layers + head)
                   for head in range(self.n_mtp_heads))

    def block_bytes(self, index: int) -> int:
        """Per-token weights of block ``index``, MTP heads counted as blocks."""
        if index < self.moe.n_dense_layers:
            return self.dense_layer_bytes(index)
        if index < self.n_layers:
            return self.hot_bytes_per_moe_layer(index)
        return self.mtp_hot_bytes(index - self.n_layers)

    def engram_bytes(self) -> int:
        """All conditional-memory tables, packed at the weight format."""
        if self.engram is None:
            return 0
        return self.engram.total_bytes(self.hidden_size, self.weights)

    def vision_bytes(self) -> int:
        if self.vision is None:
            return 0
        return self.vision.total_bytes(self.hidden_size, self.weights)

    def total_bytes(self) -> int:
        """Everything the fleet has to store — backbone, draft blocks, Engram
        tables, and the vision tower."""
        weights = (self.embedding_bytes() + self.lm_head_bytes()
                   + self._final_head_bytes() + self.mtp_bytes()
                   + self.engram_bytes() + self.vision_bytes())
        for index in range(self.n_layers):
            if index < self.moe.n_dense_layers:
                weights += self.dense_layer_bytes(index)
            else:
                weights += (self.hot_bytes_per_moe_layer(index)
                            + self.cold_bytes_per_moe_layer(index))
        return weights

    def total_params(self, include_mtp: bool = True) -> int:
        """Parameter count, for checking a profile against a model card.

        Counts the backbone blocks plus, when ``include_mtp`` is on, the draft
        blocks — matching the card convention of quoting the backbone and
        leaving auxiliary stores (Engram tables, the vision tower) to their own
        lines. DeepSeek's published totals exclude the MTP head, so
        ``include_mtp`` has to be off to compare against a card and on to size
        a fleet, which does have to store it.
        """
        params = 2 * self.vocab_size * self.hidden_size
        params += self._final_head_params()
        blocks = self.planning_layers if include_mtp else self.n_layers
        for index in range(blocks):
            moe = self.moe_for_block(index)
            per_expert = moe.expert_params() * self.hidden_size
            attn = self.attention.weight_params(self.hidden_size, index)
            params += attn
            # input and post-attention RMSNorms
            params += 2 * self.hidden_size
            if self.hc_mult > 1:
                params += self._hc_params()
            if index < moe.n_dense_layers and index < self.n_layers:
                params += moe.dense_mlp_params(self.hidden_size)
                continue
            params += self.hidden_size * moe.n_routed_experts
            if index < moe.n_hash_layers:
                params += self.vocab_size * moe.top_k
            else:
                params += moe.n_routed_experts
            params += per_expert * (moe.n_routed_experts
                                    + moe.n_shared_experts)
            if index >= self.n_layers:
                params += 2 * self.hidden_size * self.hidden_size
                params += self._mtp_extra_params()
                params += self.draft_markov_rank * self.hidden_size
        return int(params)

    def activated_params(self) -> int:
        """Parameters a single token actually multiplies against.

        MTP heads are not counted: they run only when speculative decoding is
        on, which is what the published activated-parameter figures assume.
        """
        per_expert = self.moe.expert_params() * self.hidden_size
        params = float(self.vocab_size * self.hidden_size)
        for index in range(self.n_layers):
            attn = self.attention.weight_params(self.hidden_size, index)
            if index < self.moe.n_dense_layers:
                params += attn + self.moe.dense_mlp_params(self.hidden_size)
                continue
            params += (attn + self.hidden_size * self.moe.n_routed_experts
                       + per_expert * (self.moe.top_k
                                       + self.moe.n_shared_experts))
        return int(params)

    def kv_cache_bytes_for_layer(self, layer: int, context_tokens: int,
                                 dtype: str = "fp8") -> int:
        """Cache one layer holds for one sequence at this context length."""
        per_token = self.attention.kv_cache_bytes_per_token(dtype, layer)
        return context_tokens * per_token + self.attention.state_bytes(dtype,
                                                                       layer)

    def kv_read_bytes_for_layer(self, layer: int, context_tokens: int,
                                dtype: str = "fp8") -> int:
        """Cache one layer *reads* to decode one token.

        Equal to what it holds under dense attention, and far less under CSA.
        """
        return self.attention.kv_read_bytes(context_tokens, dtype, layer)

    def kv_cache_bytes(self, context_tokens: int, dtype: str = "fp8") -> int:
        return sum(self.kv_cache_bytes_for_layer(layer, context_tokens, dtype)
                   for layer in range(self.n_layers))

    def block_kv_bytes(self, index: int, context_tokens: int,
                       dtype: str = "fp8") -> int:
        """Cache block ``index`` holds, MTP heads included.

        An MTP head runs its own attention and so keeps its own cache; it is
        one more block, sized like the layer whose position it occupies.
        """
        return self.kv_cache_bytes_for_layer(index, context_tokens, dtype)

    def block_kv_read_bytes(self, index: int, context_tokens: int,
                            dtype: str = "fp8") -> int:
        return self.kv_read_bytes_for_layer(index, context_tokens, dtype)

    def activation_bytes(self, dtype: str = "bf16") -> int:
        """One token's residual state on the wire between pipeline stages.

        With hyper-connections this is the whole ``hc_mult``-wide stream, not
        the hidden-wide layer input: the next block's residual mixing needs
        every branch, so a shelf boundary cannot ship only the part the layer
        consumes.
        """
        return int(round(self.hc_mult * self.hidden_size * DTYPE_BYTES[dtype]))

    def layer_kinds(self) -> List[str]:
        return [self.attention.kind(i) for i in range(self.planning_layers)]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

#: DeepSeek-V3 / R1, from the published config: 671 B total / 37 B activated,
#: 61 layers, hidden 7168, 256 routed + 1 shared expert per MoE layer, top-8
#: within 4 of 8 groups, first 3 layers dense, MLA with a 512-wide latent, one
#: MTP head, native FP8.
DEEPSEEK_V3 = ModelProfile(
    name="deepseek-v3",
    n_layers=61,
    hidden_size=7168,
    vocab_size=129280,
    attention=MLAConfig(n_heads=128, kv_lora_rank=512, q_lora_rank=1536,
                        qk_nope_head_dim=128, qk_rope_head_dim=64,
                        v_head_dim=128),
    moe=MoEConfig(n_routed_experts=256, n_shared_experts=1, top_k=8,
                  moe_intermediate_size=2048, n_dense_layers=3,
                  dense_intermediate_size=18432, n_group=8, topk_group=4),
    weights=FP8_BLOCK128,
    n_mtp_heads=1,
    source="deepseek-ai/DeepSeek-V3 config.json",
)

#: Per-block attention schedule of DeepSeek-V4-Pro: HCA for the first two
#: blocks, then CSA and HCA interleaved, with the trailing entry describing the
#: MTP block. Taken verbatim from the checkpoint's ``compress_ratios``.
_V4_PRO_RATIOS: Tuple[int, ...] = tuple(
    [128, 128] + [4, 128] * 29 + [4, 0])

#: Per-block attention schedule of DeepSeek-V4-Flash: two pure sliding-window
#: blocks, then CSA and HCA interleaved, then the MTP block.
_V4_FLASH_RATIOS: Tuple[int, ...] = tuple(
    [0, 0] + [4, 128] * 20 + [4, 0])

#: DeepSeek-V4-Pro, from the published config: 1,599 B total
#: (1,573 B excluding the MTP head) / 49 B activated, 61 layers, hidden 7168,
#: all-MoE with 384 routed + 1 shared expert, top-6, hash gating on the first
#: three layers, hybrid CSA (m=4, top-1024) and HCA (m'=128) attention with 128
#: query heads of width 512, one MTP head, hyper-connection width 4. Expert
#: weights are MXFP4 and everything else is FP8.
DEEPSEEK_V4_PRO = ModelProfile(
    name="deepseek-v4-pro",
    n_layers=61,
    hidden_size=7168,
    vocab_size=129280,
    attention=HybridAttentionConfig(
        n_heads=128, head_dim=512, q_lora_rank=1536,
        o_groups=16, o_group_dim=1024, qk_rope_head_dim=64,
        index_n_heads=64, index_head_dim=128, index_topk=1024,
        sliding_window=128, csa_ratio=4, hca_ratio=128,
        compress_ratios=_V4_PRO_RATIOS),
    moe=MoEConfig(n_routed_experts=384, n_shared_experts=1, top_k=6,
                  moe_intermediate_size=3072, n_dense_layers=0,
                  dense_intermediate_size=0, n_hash_layers=3),
    weights=FP8_UE8M0,
    expert_weights=MXFP4,
    n_mtp_heads=1,
    hc_mult=4,
    source="deepseek-ai/DeepSeek-V4-Pro config.json; arXiv:2606.19348 §4.2.1",
)

#: DeepSeek-V4-Flash, from the published config: 291 B total
#: (284 B excluding the MTP head) / 13 B activated, 43 layers, hidden 4096,
#: 256 routed + 1 shared expert, top-6, the same hybrid
#: attention with 64 query heads and a 512-entry indexer top-k, and pure
#: sliding-window attention in the first two blocks. Same mixed FP4/FP8 weights
#: as Pro, at a fifth of the size — which is the whole reason to care about it
#: here: it is the first member of the family a small fleet can hold.
DEEPSEEK_V4_FLASH = ModelProfile(
    name="deepseek-v4-flash",
    n_layers=43,
    hidden_size=4096,
    vocab_size=129280,
    attention=HybridAttentionConfig(
        n_heads=64, head_dim=512, q_lora_rank=1024,
        o_groups=8, o_group_dim=1024, qk_rope_head_dim=64,
        index_n_heads=64, index_head_dim=128, index_topk=512,
        sliding_window=128, csa_ratio=4, hca_ratio=128,
        compress_ratios=_V4_FLASH_RATIOS),
    moe=MoEConfig(n_routed_experts=256, n_shared_experts=1, top_k=6,
                  moe_intermediate_size=2048, n_dense_layers=0,
                  dense_intermediate_size=0, n_hash_layers=3),
    weights=FP8_UE8M0,
    expert_weights=MXFP4,
    n_mtp_heads=1,
    hc_mult=4,
    source="deepseek-ai/DeepSeek-V4-Flash config.json; arXiv:2606.19348 §4.2.1",
)

#: Per-block attention schedule of DeepSeek-V4.1-Flash, verbatim from the
#: checkpoint's ``compress_ratios``: two sliding-window blocks, eighteen
#: compressed encoder blocks at m=2, twenty decoder blocks that write no
#: stream of their own (entry 1), then the three DSpark draft blocks.
_V4_1_FLASH_RATIOS: Tuple[int, ...] = tuple(
    [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0])

#: DeepSeek-V4.1-Flash, from the published config: 552 B backbone
#: / 16 B activated on decode (8 B on prefill — the CED split makes the
#: decoder's cache come pre-built), plus ~196 B of Engram tables and the
#: vision tower that the card reports separately. 40 layers as a 20-layer
#: causal encoder + 20-layer decoder, hidden 5120, all-MoE with 384 routed +
#: 1 shared expert, top-6, CSA2 attention whose whole growing KV cache lives
#: on the few source layers (4 main-KV streams + 8 indexer streams, all
#: FP4 at m=2 — 864 B/token derived vs the card's ~890), three DSpark draft
#: blocks with their own 128-expert top-3 MoE, hyper-connection width 4.
#: Weights are FP8 at the finer 32x32 tile; experts stay MXFP4.
DEEPSEEK_V4_1_FLASH = ModelProfile(
    name="deepseek-v4.1-flash",
    n_layers=40,
    hidden_size=5120,
    vocab_size=129280,
    attention=CSA2Config(
        n_heads=64, head_dim=512, q_lora_rank=1280,
        o_lora_rank=1024, o_groups=8, qk_rope_head_dim=64,
        index_n_heads=32, index_head_dim=128, index_topk=512,
        sliding_window=128, csa_ratio=2,
        compress_ratios=_V4_1_FLASH_RATIOS,
        kv_source_layers=(2, 8, 14, 20),
        index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
        candidate_source_layer=20,
        candidate_topk_blocks=2048, candidate_block_size=8,
        n_encoder_layers=20),
    moe=MoEConfig(n_routed_experts=384, n_shared_experts=1, top_k=6,
                  moe_intermediate_size=2304, n_dense_layers=0,
                  dense_intermediate_size=0),
    draft_moe=MoEConfig(n_routed_experts=128, n_shared_experts=1, top_k=3,
                        moe_intermediate_size=2304, n_dense_layers=0,
                        dense_intermediate_size=0),
    draft_markov_rank=256,
    engram=EngramConfig(
        layer_ids=(1, 14),
        num_embeddings=(384006168, 384016682),
        n_heads=8, head_dim=256,
        vocab_size=16000000, compressed_vocab_size=99092,
        max_ngram_size=4),
    vision=VisionConfig(n_layers=32, hidden_size=1024, n_heads=16,
                        intermediate_size=2816, patch_size=14,
                        downsample_ratio=3),
    weights=FP8_UE8M0_32,
    expert_weights=MXFP4,
    n_mtp_heads=3,
    hc_mult=4,
    source="deepseek-ai/DeepSeek-V4.1-Flash config.json",
)

#: A small MoE with the same architecture, for a fleet of two or three consoles
#: and for CI. Not a real checkpoint; a shape.
DEEPSEEK_TINY = ModelProfile(
    name="deepseek-tiny",
    n_layers=8,
    hidden_size=1024,
    vocab_size=32768,
    attention=MLAConfig(n_heads=16, kv_lora_rank=128, q_lora_rank=None,
                        qk_nope_head_dim=64, qk_rope_head_dim=32,
                        v_head_dim=64),
    moe=MoEConfig(n_routed_experts=32, n_shared_experts=1, top_k=4,
                  moe_intermediate_size=512, n_dense_layers=1,
                  dense_intermediate_size=2048, n_group=4, topk_group=2),
    weights=FP8_BLOCK128,
    n_mtp_heads=0,
)

PROFILES: Dict[str, ModelProfile] = {
    p.name: p for p in (DEEPSEEK_V3, DEEPSEEK_V4_PRO, DEEPSEEK_V4_FLASH,
                        DEEPSEEK_V4_1_FLASH, DEEPSEEK_TINY)
}


def profile_for(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown model profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def _attention_summary(profile: ModelProfile) -> str:
    kinds = profile.layer_kinds()
    counts: Dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return ", ".join(f"{n} {kind.upper()}"
                     for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def describe(profile: ModelProfile) -> List[str]:
    """Human-readable size breakdown, shared by the CLIs."""
    gib = 1024 ** 3
    mib = 1024 ** 2
    fmt = profile.weights.dtype
    if profile.mixed_precision:
        fmt = f"{profile.expert_quant.dtype} experts / {fmt} elsewhere"
    n_encoder = getattr(profile.attention, "n_encoder_layers", 0)
    structure = f"{profile.n_layers}"
    if n_encoder:
        structure += (f" ({n_encoder} causal-encoder + "
                      f"{profile.n_layers - n_encoder} decoder)")
    lines = [
        f"{profile.name}  ({fmt}"
        + (", ASSUMED CONFIGURATION" if profile.assumed else "") + ")",
        f"  layers                {structure} "
        f"({profile.moe.n_dense_layers} dense, {profile.n_moe_layers} MoE)",
        f"  attention             {_attention_summary(profile)}",
        f"  hidden                {profile.hidden_size}"
        + (f" (residual {profile.hc_mult}x wide)" if profile.hc_mult > 1
           else ""),
        f"  routed experts/layer  {profile.moe.n_routed_experts} "
        f"(top-{profile.moe.top_k} of {profile.moe.topk_group}/"
        f"{profile.moe.n_group} groups)",
        f"  params total          {profile.total_params() / 1e9:.1f} B",
        f"  params activated      {profile.activated_params() / 1e9:.1f} B",
        f"  weights total         {profile.total_bytes() / gib:.1f} GiB",
        f"  one routed expert     {profile.expert_bytes() / mib:.2f} MiB",
        f"  hot per MoE layer     "
        f"{profile.hot_bytes_per_moe_layer() / mib:.1f} MiB",
        f"  cold per MoE layer    "
        f"{profile.cold_bytes_per_moe_layer() / gib:.2f} GiB",
        f"  KV cache @ 8k ctx     "
        f"{profile.kv_cache_bytes(8192) / mib:.0f} MiB",
    ]
    if profile.draft_moe is not None:
        lines.append(
            f"  draft blocks          {profile.n_mtp_heads} "
            f"({profile.draft_moe.n_routed_experts} experts, "
            f"top-{profile.draft_moe.top_k})")
    if profile.engram is not None:
        lines.append(
            f"  engram tables         "
            f"{profile.engram.params(profile.hidden_size) / 1e9:.1f} B "
            f"({profile.engram_bytes() / gib:.1f} GiB) at layers "
            + ",".join(str(i) for i in profile.engram.layer_ids)
            + " — hash-addressed, NVMe-tier by design")
    if profile.vision is not None:
        lines.append(
            f"  vision tower          "
            f"{profile.vision.params(profile.hidden_size) / 1e9:.2f} B "
            f"({profile.vision_bytes() / mib:.0f} MiB), once per image")
    if profile.source:
        lines.append(f"  source                {profile.source}")
    if profile.assumed:
        lines.append("  assumptions:")
        lines.extend(f"    - {a}" for a in profile.assumptions)
    return lines
