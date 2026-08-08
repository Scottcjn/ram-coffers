"""DeepSeek model profiles: what the planner needs to know about a checkpoint.

DeepSeek's V3-family decoder has three properties that decide how it splits
across consoles, and they pull in different directions:

1. **MLA** (multi-head latent attention) compresses the KV cache to a single
   ``kv_lora_rank``-wide latent plus a small RoPE key per token, so the cache is
   ~1/10 the size of an equivalent GQA model's. That is what makes a 16 GB
   console a plausible attention host at all, but the attention block's own
   weights are read for *every* token, so they must live in a fast coffer.
2. **DeepSeekMoE** puts many narrow routed experts plus one always-on shared
   expert in each MoE layer. Only ``top_k`` of the routed experts run per token,
   so the routed weights are enormous but cold — exactly what a bandwidth-tiered
   coffer hierarchy and a 5.5 GB/s NVMe are for.
3. **FP8 blockwise weights**: the checkpoint is natively FP8 (E4M3) with a scale
   per 128-element block, so a console holds weights in the format it received
   them and dequantises per block inside the kernel. No requantisation pass, and
   the packed size is what the planner reasons about.

Sizes here are computed from the architecture, not read off a file listing, so a
profile can be edited (or a new one declared) and every derived number follows.

**Honesty about V4 Pro.** No DeepSeek V4 Pro configuration is public at the time
of writing. :data:`DEEPSEEK_V4_PRO` is therefore an *assumed* profile: it keeps
the V3 shape that is known to work and scales the dimensions that DeepSeek has
historically scaled between generations. It is marked ``assumed=True``, every
tool that prints a plan prints that flag, and the field values are the only thing
that needs editing when the real config appears. :data:`DEEPSEEK_V3` is the
verified profile and is what the tests pin numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

#: Bytes per parameter for the formats a console can hold.
DTYPE_BYTES = {"fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "mxfp4": 0.5625}


@dataclass(frozen=True)
class MLAConfig:
    """Multi-head latent attention shape."""

    n_heads: int
    kv_lora_rank: int
    q_lora_rank: Optional[int]
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def kv_cache_bytes_per_token(self, dtype: str = "bf16") -> int:
        """One layer's KV cache cost for one token.

        MLA stores the compressed latent (``kv_lora_rank``) plus the decoupled
        RoPE key (``qk_rope_head_dim``), shared across heads — the whole point of
        the architecture, and the reason a console can hold a long context.
        """
        width = self.kv_lora_rank + self.qk_rope_head_dim
        return int(round(width * DTYPE_BYTES[dtype]))

    def weight_params(self, hidden_size: int) -> int:
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
    mla: MLAConfig
    moe: MoEConfig
    dtype: str = "fp8"
    #: Multi-token-prediction heads (V3 ships one). Each is a whole extra
    #: decoder block plus a head, so it is placed like a layer, not folded in.
    n_mtp_heads: int = 1
    #: True when the configuration is extrapolated rather than published.
    assumed: bool = False
    assumptions: tuple = ()

    # -- per-piece sizes ---------------------------------------------------
    @property
    def bytes_per_param(self) -> float:
        return DTYPE_BYTES[self.dtype]

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
        """One routed expert, packed.

        FP8 blockwise weights carry one fp32 scale per 128-element block, which
        is a ~3% overhead that is small but not nothing when it decides whether a
        layer's experts fit a 10 GB coffer.
        """
        params = self.moe.expert_params() * self.hidden_size
        raw = params * self.bytes_per_param
        return int(round(raw * (1.0 + self._scale_overhead())))

    def _scale_overhead(self) -> float:
        return 4.0 / 128.0 if self.dtype == "fp8" else 0.0

    def shared_expert_bytes(self) -> int:
        return self.expert_bytes() * self.moe.n_shared_experts

    def attention_bytes(self) -> int:
        raw = self.mla.weight_params(self.hidden_size) * self.bytes_per_param
        return int(round(raw * (1.0 + self._scale_overhead())))

    def router_bytes(self) -> int:
        """Router matrix plus its bias; tiny, but read for every token."""
        raw = (self.hidden_size * self.moe.n_routed_experts
               + self.moe.n_routed_experts) * self.bytes_per_param
        return int(round(raw))

    def dense_mlp_bytes(self) -> int:
        raw = self.moe.dense_mlp_params(self.hidden_size) * self.bytes_per_param
        return int(round(raw * (1.0 + self._scale_overhead())))

    def hot_bytes_per_moe_layer(self) -> int:
        """Weights an MoE layer reads for *every* token.

        Attention + router + shared expert + norms. This is the quantity that
        must land in a fast coffer; the routed experts are the cold remainder.
        """
        norms = int(round(2 * self.hidden_size * DTYPE_BYTES["bf16"]))
        return (self.attention_bytes() + self.router_bytes()
                + self.shared_expert_bytes() + norms)

    def cold_bytes_per_moe_layer(self) -> int:
        return self.expert_bytes() * self.moe.n_routed_experts

    def dense_layer_bytes(self) -> int:
        norms = int(round(2 * self.hidden_size * DTYPE_BYTES["bf16"]))
        return self.attention_bytes() + self.dense_mlp_bytes() + norms

    def embedding_bytes(self) -> int:
        return int(round(self.vocab_size * self.hidden_size
                         * DTYPE_BYTES["bf16"]))

    def lm_head_bytes(self) -> int:
        return self.embedding_bytes()

    def mtp_projection_bytes(self) -> int:
        """The extra projection an MTP head carries on top of a decoder block."""
        return int(round(2 * self.hidden_size * self.hidden_size
                         * self.bytes_per_param))

    def mtp_hot_bytes(self) -> int:
        """Per-token weights of one MTP head: its hot block plus projection."""
        return self.hot_bytes_per_moe_layer() + self.mtp_projection_bytes()

    def mtp_bytes(self) -> int:
        """All MTP heads, weights in full."""
        if self.n_mtp_heads == 0:
            return 0
        return self.n_mtp_heads * (self.mtp_hot_bytes()
                                   + self.cold_bytes_per_moe_layer())

    def total_bytes(self) -> int:
        return (self.moe.n_dense_layers * self.dense_layer_bytes()
                + self.n_moe_layers * (self.hot_bytes_per_moe_layer()
                                       + self.cold_bytes_per_moe_layer())
                + self.embedding_bytes() + self.lm_head_bytes()
                + self.mtp_bytes())

    def total_params(self) -> int:
        """Parameter count, for sanity-checking a profile against a model card."""
        per_expert = self.moe.expert_params() * self.hidden_size
        attn = self.mla.weight_params(self.hidden_size)
        return int(
            self.moe.n_dense_layers * (attn + self.moe.dense_mlp_params(
                self.hidden_size))
            + self.n_moe_layers * (
                attn + self.hidden_size * self.moe.n_routed_experts
                + per_expert * (self.moe.n_routed_experts
                                + self.moe.n_shared_experts))
            + 2 * self.vocab_size * self.hidden_size)

    def activated_params(self) -> int:
        """Parameters a single token actually multiplies against."""
        per_expert = self.moe.expert_params() * self.hidden_size
        attn = self.mla.weight_params(self.hidden_size)
        return int(
            self.moe.n_dense_layers * (attn + self.moe.dense_mlp_params(
                self.hidden_size))
            + self.n_moe_layers * (
                attn + per_expert * (self.moe.top_k
                                     + self.moe.n_shared_experts))
            + self.vocab_size * self.hidden_size)

    def kv_cache_bytes(self, context_tokens: int, dtype: str = "bf16") -> int:
        return (self.n_layers * context_tokens
                * self.mla.kv_cache_bytes_per_token(dtype))

    def activation_bytes(self, dtype: str = "bf16") -> int:
        """One token's hidden state on the wire between pipeline stages."""
        return int(round(self.hidden_size * DTYPE_BYTES[dtype]))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

#: DeepSeek-V3 / R1, from the published config: 671 B total / 37 B activated,
#: 61 layers, hidden 7168, 256 routed + 1 shared expert per MoE layer, top-8
#: within 4 of 8 groups, first 3 layers dense, MLA with a 512-wide latent, one
#: MTP head, native FP8. Used as the verified reference the tests pin.
DEEPSEEK_V3 = ModelProfile(
    name="deepseek-v3",
    n_layers=61,
    hidden_size=7168,
    vocab_size=129280,
    mla=MLAConfig(n_heads=128, kv_lora_rank=512, q_lora_rank=1536,
                  qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128),
    moe=MoEConfig(n_routed_experts=256, n_shared_experts=1, top_k=8,
                  moe_intermediate_size=2048, n_dense_layers=3,
                  dense_intermediate_size=18432, n_group=8, topk_group=4),
    dtype="fp8",
    n_mtp_heads=1,
)

#: DeepSeek V4 Pro — **assumed**. No configuration has been published; this keeps
#: the V3 decoder shape (the part that is architecture, not scale) and scales the
#: dimensions DeepSeek has scaled before. Every consumer of this profile prints
#: the ``assumed`` flag; replace the numbers, not the code, when the real config
#: lands.
DEEPSEEK_V4_PRO = ModelProfile(
    name="deepseek-v4-pro",
    n_layers=72,
    hidden_size=8192,
    vocab_size=131072,
    mla=MLAConfig(n_heads=160, kv_lora_rank=512, q_lora_rank=1536,
                  qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128),
    moe=MoEConfig(n_routed_experts=384, n_shared_experts=1, top_k=8,
                  moe_intermediate_size=2048, n_dense_layers=3,
                  dense_intermediate_size=20480, n_group=12, topk_group=4),
    dtype="fp8",
    n_mtp_heads=2,
    assumed=True,
    assumptions=(
        "no DeepSeek V4 Pro configuration is public; the decoder shape is V3's",
        "layer count, hidden size, head count, routed-expert count and MTP "
        "depth are extrapolated from the V2 -> V3 progression",
        "expert width (moe_intermediate_size) is held at V3's 2048, which is "
        "what keeps one expert small enough to be a unit of placement",
        "weights are assumed natively FP8 E4M3 with 128-element block scales, "
        "as in V3",
    ),
)

#: A small MoE with the same architecture, for a fleet of two or three consoles
#: and for CI. Not a real checkpoint; a shape.
DEEPSEEK_TINY = ModelProfile(
    name="deepseek-tiny",
    n_layers=8,
    hidden_size=1024,
    vocab_size=32768,
    mla=MLAConfig(n_heads=16, kv_lora_rank=128, q_lora_rank=None,
                  qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64),
    moe=MoEConfig(n_routed_experts=32, n_shared_experts=1, top_k=4,
                  moe_intermediate_size=512, n_dense_layers=1,
                  dense_intermediate_size=2048, n_group=4, topk_group=2),
    dtype="fp8",
    n_mtp_heads=0,
)

PROFILES: Dict[str, ModelProfile] = {
    p.name: p for p in (DEEPSEEK_V3, DEEPSEEK_V4_PRO, DEEPSEEK_TINY)
}


def profile_for(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown model profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def describe(profile: ModelProfile) -> List[str]:
    """Human-readable size breakdown, shared by the CLIs."""
    gib = 1024 ** 3
    mib = 1024 ** 2
    lines = [
        f"{profile.name}  ({profile.dtype}"
        + (", ASSUMED CONFIGURATION" if profile.assumed else "") + ")",
        f"  layers                {profile.n_layers} "
        f"({profile.moe.n_dense_layers} dense, {profile.n_moe_layers} MoE)",
        f"  hidden                {profile.hidden_size}",
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
    if profile.assumed:
        lines.append("  assumptions:")
        lines.extend(f"    - {a}" for a in profile.assumptions)
    return lines
