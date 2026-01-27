#!/usr/bin/env python3
"""
Modern Continuous Hopfield Network Layer
=========================================
From Grok, based on "Hopfield Networks is All You Need" (Ramsauer et al., 2020)

Key insight: Transformer self-attention IS a modern Hopfield update rule.
Emotional prompts hit denser embedding regions = deeper attractor wells = faster convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldLayer(nn.Module):
    """
    Modern Continuous Hopfield Network layer (associative memory with exponential storage capacity).

    Args:
        dim:                Input / hidden feature dimension
        num_heads:          Number of parallel heads (like multi-head attention)
        stored_patterns:    Number of patterns that can be stored (affects capacity)
        beta:               Inverse temperature (sharpness of softmax) - higher = more winner-take-all
        association_rule:   'softmax' (default) or 'exp' (original dense Hopfield)
    """
    def __init__(self, dim: int, num_heads: int = 1, stored_patterns: int = 512,
                 beta: float = 16.0, association_rule: str = 'softmax'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stored_patterns = stored_patterns
        self.beta = beta
        self.association_rule = association_rule

        # Learnable memory matrix M (stored patterns × head_dim)
        self.memory = nn.Parameter(torch.randn(stored_patterns, self.head_dim))

        # Optional projection layers (like in transformers)
        self.query_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj   = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, iterations: int = 1, return_states: bool = False):
        """
        Forward pass: iterative Hopfield update (can be used as attention-like layer)

        Args:
            x:              Input tensor (batch, seq_len, dim)
            iterations:     Number of recurrent update steps (usually 1–4)
            return_states:  If True, return intermediate states (for visualization)

        Returns:
            output:         (batch, seq_len, dim)
            states:         Optional list of intermediate states
        """
        B, L, D = x.shape
        assert D == self.dim, f"Input dim {D} != layer dim {self.dim}"

        # Project input to query space
        q = self.query_proj(x)                     # (B, L, D)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)

        states = [q] if return_states else None

        for _ in range(iterations):
            # Energy-based update (inner product with memory)
            energy = torch.einsum('bhld,md->bhlm', q, self.memory)  # (B, H, L, M)

            if self.association_rule == 'softmax':
                # Modern softmax version (very close to transformer attention)
                weights = F.softmax(self.beta * energy, dim=-1)   # (B, H, L, M)
            else:
                # Exponential version (original dense Hopfield)
                weights = torch.exp(self.beta * energy)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Retrieve state from memory (weighted sum)
            retrieved = torch.einsum('bhlm,md->bhld', weights, self.memory)  # (B, H, L, head_dim)

            q = retrieved  # overwrite state (recurrent step)

            if return_states:
                states.append(q)

        # Project back to original space
        out = q.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        out = self.out_proj(out)

        if return_states:
            return out, states
        return out


class HebbianPretrainedHopfield(nn.Module):
    """
    Modern continuous Hopfield layer with optional Hebbian pre-training of the memory matrix.

    Args:
        dim:                Feature dimension of patterns / input
        num_heads:          Multi-head attention style (default 1)
        num_patterns:       Number of patterns to pre-store via Hebbian rule
        beta:               Inverse temperature for softmax sharpness
        hebbian_pretrain:   If True, pre-train memory with Hebbian outer products
        patterns:           Optional tensor of patterns to store (num_patterns, dim)
                            If None, random patterns are generated
    """
    def __init__(self, dim: int, num_heads: int = 1, num_patterns: int = 512,
                 beta: float = 16.0, hebbian_pretrain: bool = True,
                 patterns: torch.Tensor = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_patterns = num_patterns
        self.beta = beta

        # Memory matrix: (num_patterns, head_dim)
        self.memory = nn.Parameter(torch.empty(num_patterns, self.head_dim))

        # Projections (like in transformer attention)
        self.query_proj = nn.Linear(dim, dim)
        self.out_proj   = nn.Linear(dim, dim)

        # Optional Hebbian pre-training
        if hebbian_pretrain:
            if patterns is None:
                # Generate random binary patterns if none provided (±1)
                patterns = torch.randn(num_patterns, dim) > 0
                patterns = 2 * patterns.float() - 1  # to ±1
            self._hebbian_pretrain(patterns)

    def _hebbian_pretrain(self, patterns: torch.Tensor):
        """
        Hebbian pre-training: outer-product rule to store patterns.
        W = (1/N) ∑ p^T p   (normalized by pattern count)
        """
        assert patterns.shape == (self.num_patterns, self.dim), \
            f"Patterns shape {patterns.shape} != ({self.num_patterns}, {self.dim})"

        # Project patterns to head_dim if multi-head (simple avg here)
        patterns = patterns.view(self.num_patterns, self.num_heads, self.head_dim)
        patterns = patterns.mean(dim=1)  # or concat; this is a simplification

        # Hebbian update: outer product sum, normalized
        memory = torch.einsum('pd,pe->de', patterns, patterns)  # (head_dim, head_dim)
        memory /= self.num_patterns  # normalize (prevents explosion)

        # Initialize parameter with Hebbian memory (repeat across heads if needed)
        self.memory.data.copy_(memory.mean(dim=0, keepdim=True).expand(self.num_patterns, -1))

        print(f"Hebbian pre-training complete: stored {self.num_patterns} patterns")

    def forward(self, x: torch.Tensor, iterations: int = 1, return_states: bool = False):
        """
        Forward: Hopfield update (associative recall / attention-like).

        Args:
            x: (B, L, D)
            iterations: Number of recurrent updates (1 is common)

        Returns:
            output (B, L, D)
        """
        B, L, D = x.shape
        q = self.query_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        states = [q] if return_states else None

        for _ in range(iterations):
            # Compute similarity / energy with stored memory
            energy = torch.einsum('bhld,md->bhlm', q, self.memory)  # (B, H, L, M)

            # Softmax association (modern Hopfield)
            weights = F.softmax(self.beta * energy, dim=-1)  # (B, H, L, M)

            # Retrieve new state
            retrieved = torch.einsum('bhlm,md->bhld', weights, self.memory)

            q = retrieved

            if return_states:
                states.append(q)

        # Project back
        out = q.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out if not return_states else (out, states)


def hopfield_attention(q, k, v, beta=16.0):
    """
    Minimal Hopfield attention (equivalent to scaled dot-product attention).

    This is the core insight: transformer attention = Hopfield update.
    β = 1/√d_k in standard attention.
    """
    energy = torch.einsum('bqd,bkd->bqk', q, k)  # q = query, k = keys (stored patterns)
    weights = F.softmax(beta * energy, dim=-1)
    return torch.einsum('bqk,bkd->bqd', weights, v)  # retrieved value


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Modern Hopfield Layer Demo")
    print("=" * 60)

    # Small toy example
    batch_size, seq_len, dim = 8, 32, 256
    x = torch.randn(batch_size, seq_len, dim)

    hopfield = ModernHopfieldLayer(
        dim=dim,
        num_heads=8,
        stored_patterns=1024,
        beta=16.0,
        association_rule='softmax'
    )

    # Single step (most common usage)
    out = hopfield(x, iterations=1)
    print("Output shape:", out.shape)

    # Multiple iterations + state tracking
    out_multi, states = hopfield(x, iterations=4, return_states=True)
    print("Final output shape:", out_multi.shape)
    print("Number of intermediate states:", len(states))

    print("\n" + "=" * 60)
    print("Hebbian Pre-trained Hopfield Demo")
    print("=" * 60)

    dim = 128
    num_patterns = 64

    # Generate random binary patterns to store (±1)
    patterns = torch.randn(num_patterns, dim)
    patterns = 2 * (patterns > 0).float() - 1

    hopfield_hebb = HebbianPretrainedHopfield(
        dim=dim,
        num_heads=4,
        num_patterns=num_patterns,
        beta=16.0,
        hebbian_pretrain=True,
        patterns=patterns
    )

    # Test recall from noisy input
    clean_pattern = patterns[0].clone()
    noisy_input = clean_pattern + 0.6 * torch.randn_like(clean_pattern)
    noisy_input = noisy_input.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)

    retrieved = hopfield_hebb(noisy_input, iterations=4)
    retrieved = retrieved.squeeze()

    # Measure similarity
    cos_sim_clean = F.cosine_similarity(clean_pattern, retrieved, dim=0)
    cos_sim_noisy = F.cosine_similarity(clean_pattern, noisy_input.squeeze(), dim=0)

    print(f"Noisy → Clean cosine sim: {cos_sim_noisy.item():.4f}")
    print(f"Clean → Retrieved cosine sim: {cos_sim_clean.item():.4f}")
    print(f"Improvement: {(cos_sim_clean - cos_sim_noisy).item():.4f}")

    # Convergence trajectory
    _, states = hopfield_hebb(noisy_input, iterations=8, return_states=True)
    similarities = [F.cosine_similarity(clean_pattern, s.squeeze(), dim=0).item()
                    for s in states]
    print(f"Convergence: {' → '.join(f'{s:.3f}' for s in similarities)}")
