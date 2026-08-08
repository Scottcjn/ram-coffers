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
#: OCP MXFP4: E2M1 values with one E8M0 scale per 32-element tile.
MXFP4 = QuantSpec("mxfp4", scale_bytes=1.0, scale_block=32)


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

    def kv_cache_bytes_per_token(self, dtype: str = "bf16",
                                 layer: int = 0) -> int:
        """One layer's KV cache cost for one token.

        MLA stores the compressed latent (``kv_lora_rank``) plus the decoupled
        RoPE key (``qk_rope_head_dim``), shared across heads — the whole point of
        the architecture, and the reason a console can hold a long context.
        """
        width = self.kv_lora_rank + self.qk_rope_head_dim
        return int(round(width * DTYPE_BYTES[dtype]))

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
        kind = self.kind(layer)
        # Queries: down to d_c, then up to one c-wide query per head.
        q = d * self.q_lora_rank + self.q_lora_rank * nh * c
        # Grouped output: heads within a group share an intermediate of width
        # d_g, and each group projects back to d.
        out = nh * c * self.o_group_dim + self.o_groups * self.o_group_dim * d
        if kind == "csa":
            # Two overlapping KV series and their compression weights, plus the
            # learnable positional bias each compression window carries.
            kv = 4 * d * c + 2 * self.csa_ratio * c
            indexer = (d * self.index_n_heads * self.index_head_dim
                       + d * self.index_head_dim + self.index_n_heads)
        elif kind == "hca":
            kv = 2 * d * c + self.hca_ratio * c
            indexer = 0
        else:
            # No compression: one shared KV entry straight off the hidden state.
            kv = d * c
            indexer = 0
        return q + kv + out + indexer

    def kv_cache_bytes_per_token(self, dtype: str = "bf16",
                                 layer: int = 0) -> int:
        kind = self.kind(layer)
        if kind == "swa":
            return 0  # the window is a fixed-size state, see state_bytes
        ratio = self.csa_ratio if kind == "csa" else self.hca_ratio
        entry = self.head_dim * DTYPE_BYTES[dtype] / ratio
        if kind == "csa":
            entry += (self.index_head_dim
                      * self.index_quant.bytes_per_param / ratio)
        return int(round(entry))

    def state_bytes(self, dtype: str = "bf16", layer: int = 0) -> int:
        """The sliding-window branch, which every layer carries.

        Uncompressed tail tokens live here too; both are bounded, which is why
        DeepSeek's own serving stack treats them as a fixed pool rather than as
        part of the growing cache.
        """
        return int(round(self.sliding_window * self.head_dim
                         * DTYPE_BYTES[dtype]))

    def kv_read_bytes(self, context_tokens: int, dtype: str = "bf16",
                      layer: int = 0) -> int:
        kind = self.kind(layer)
        state = self.state_bytes(dtype, layer)
        if kind == "swa":
            return state
        if kind == "hca":
            entries = context_tokens / self.hca_ratio
            return int(round(entries * self.head_dim * DTYPE_BYTES[dtype])
                       + state)
        # CSA scans every compressed entry with the FP4 indexer, then reads only
        # the top-k it selected.
        entries = context_tokens / self.csa_ratio
        scan = (entries * self.index_head_dim
                * self.index_quant.bytes_per_param)
        selected = min(entries, float(self.index_topk))
        return int(round(scan + selected * self.head_dim * DTYPE_BYTES[dtype])
                   + state)


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
    #: Multi-token-prediction heads (V3 ships one). Each is a whole extra
    #: decoder block plus a head, so it is placed like a layer, not folded in.
    n_mtp_heads: int = 1
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

    def expert_bytes(self) -> int:
        """One routed expert, packed, block scales included."""
        params = self.moe.expert_params() * self.hidden_size
        return int(round(params * self.expert_quant.bytes_per_param))

    def shared_expert_bytes(self) -> int:
        return self.expert_bytes() * self.moe.n_shared_experts

    def attention_bytes(self, layer: int = 0) -> int:
        params = self.attention.weight_params(self.hidden_size, layer)
        return int(round(params * self.bytes_per_param))

    def router_bytes(self) -> int:
        """Router matrix plus its bias; tiny, but read for every token."""
        raw = (self.hidden_size * self.moe.n_routed_experts
               + self.moe.n_routed_experts) * self.bytes_per_param
        return int(round(raw))

    def dense_mlp_bytes(self) -> int:
        raw = self.moe.dense_mlp_params(self.hidden_size) * self.bytes_per_param
        return int(round(raw))

    def hot_bytes_per_moe_layer(self, layer: int = 0) -> int:
        """Weights an MoE layer reads for *every* token.

        Attention + router + shared expert + norms. This is the quantity that
        must land in a fast coffer; the routed experts are the cold remainder.
        The hyper-connection mappings are generated from the hidden state and
        come to well under a tenth of a percent of this, so they are left out
        rather than guessed at.
        """
        norms = int(round(2 * self.hidden_size * DTYPE_BYTES["bf16"]))
        return (self.attention_bytes(layer) + self.router_bytes()
                + self.shared_expert_bytes() + norms)

    def cold_bytes_per_moe_layer(self) -> int:
        return self.expert_bytes() * self.moe.n_routed_experts

    def dense_layer_bytes(self, layer: int = 0) -> int:
        norms = int(round(2 * self.hidden_size * DTYPE_BYTES["bf16"]))
        return self.attention_bytes(layer) + self.dense_mlp_bytes() + norms

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
        """Per-token weights of one MTP head: its hot block plus projection."""
        return (self.hot_bytes_per_moe_layer(self.n_layers + head)
                + self.mtp_projection_bytes())

    def mtp_bytes(self) -> int:
        """All MTP heads, weights in full."""
        if self.n_mtp_heads == 0:
            return 0
        return sum(self.mtp_hot_bytes(head) + self.cold_bytes_per_moe_layer()
                   for head in range(self.n_mtp_heads))

    def block_bytes(self, index: int) -> int:
        """Per-token weights of block ``index``, MTP heads counted as blocks."""
        if index < self.moe.n_dense_layers:
            return self.dense_layer_bytes(index)
        if index < self.n_layers:
            return self.hot_bytes_per_moe_layer(index)
        return self.mtp_hot_bytes(index - self.n_layers)

    def total_bytes(self) -> int:
        weights = self.embedding_bytes() + self.lm_head_bytes() + self.mtp_bytes()
        for index in range(self.n_layers):
            if index < self.moe.n_dense_layers:
                weights += self.dense_layer_bytes(index)
            else:
                weights += (self.hot_bytes_per_moe_layer(index)
                            + self.cold_bytes_per_moe_layer())
        return weights

    def total_params(self, include_mtp: bool = True) -> int:
        """Parameter count, for checking a profile against a model card.

        DeepSeek's published totals exclude the MTP head, so ``include_mtp``
        has to be off to compare against a card and on to size a fleet, which
        does have to store it.
        """
        per_expert = self.moe.expert_params() * self.hidden_size
        params = 2 * self.vocab_size * self.hidden_size
        blocks = self.planning_layers if include_mtp else self.n_layers
        for index in range(blocks):
            attn = self.attention.weight_params(self.hidden_size, index)
            if index < self.moe.n_dense_layers:
                params += attn + self.moe.dense_mlp_params(self.hidden_size)
                continue
            params += (attn + self.hidden_size * self.moe.n_routed_experts
                       + per_expert * (self.moe.n_routed_experts
                                       + self.moe.n_shared_experts))
            if index >= self.n_layers:
                params += 2 * self.hidden_size * self.hidden_size
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
            params += attn + per_expert * (self.moe.top_k
                                           + self.moe.n_shared_experts)
        return int(params)

    def kv_cache_bytes_for_layer(self, layer: int, context_tokens: int,
                                 dtype: str = "bf16") -> int:
        """Cache one layer holds for one sequence at this context length."""
        per_token = self.attention.kv_cache_bytes_per_token(dtype, layer)
        return context_tokens * per_token + self.attention.state_bytes(dtype,
                                                                       layer)

    def kv_read_bytes_for_layer(self, layer: int, context_tokens: int,
                                dtype: str = "bf16") -> int:
        """Cache one layer *reads* to decode one token.

        Equal to what it holds under dense attention, and far less under CSA.
        """
        return self.attention.kv_read_bytes(context_tokens, dtype, layer)

    def kv_cache_bytes(self, context_tokens: int, dtype: str = "bf16") -> int:
        return sum(self.kv_cache_bytes_for_layer(layer, context_tokens, dtype)
                   for layer in range(self.n_layers))

    def block_kv_bytes(self, index: int, context_tokens: int,
                       dtype: str = "bf16") -> int:
        """Cache block ``index`` holds, MTP heads included.

        An MTP head runs its own attention and so keeps its own cache; it is
        one more block, sized like the layer whose position it occupies.
        """
        return self.kv_cache_bytes_for_layer(index, context_tokens, dtype)

    def block_kv_read_bytes(self, index: int, context_tokens: int,
                            dtype: str = "bf16") -> int:
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

#: DeepSeek-V4-Pro, from the published config: 1.6 T total / 49 B activated, 61
#: layers, hidden 7168, all-MoE with 384 routed + 1 shared expert, top-6, hash
#: gating on the first three layers, hybrid CSA (m=4, top-1024) and HCA (m'=128)
#: attention with 128 query heads of width 512, one MTP head, hyper-connection
#: width 4. Expert weights are MXFP4 and everything else is FP8.
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

#: DeepSeek-V4-Flash, from the published config: 284 B total / 13 B activated,
#: 43 layers, hidden 4096, 256 routed + 1 shared expert, top-6, the same hybrid
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
                        DEEPSEEK_TINY)
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
    lines = [
        f"{profile.name}  ({fmt}"
        + (", ASSUMED CONFIGURATION" if profile.assumed else "") + ")",
        f"  layers                {profile.n_layers} "
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
    if profile.source:
        lines.append(f"  source                {profile.source}")
    if profile.assumed:
        lines.append("  assumptions:")
        lines.extend(f"    - {a}" for a in profile.assumptions)
    return lines
