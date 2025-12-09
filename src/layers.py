"""
Part 2: TRM Layers
RoPE-enabled Attention + SwiGLU FFN (Qwen-compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Same settings as Qwen for compatibility (theta=1000000).
    """
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        # Return shape: [1, 1, S, head_dim] for broadcasting with [B, num_heads, S, head_dim]
        return (
            self.cos_cached[:seq_len].to(x.dtype)[None, None, :, :],
            self.sin_cached[:seq_len].to(x.dtype)[None, None, :, :]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    Simplified Qwen2-style implementation using broadcasting.

    Args:
        q, k: [B, num_heads, S, head_dim]
        cos, sin: [1, 1, S, head_dim] (pre-shaped for broadcasting)

    Returns:
        q_embed, k_embed: RoPE-applied tensors [B, num_heads, S, head_dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU Activation: FFN(x) = (SiLU(xW1) * xW2) W3
    Same as Qwen's MLP structure.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TRMAttention(nn.Module):
    """
    RoPE-enabled Causal Self-Attention (Qwen-compatible).
    head_dim=128 matches Qwen's head dimension for RoPE frequency compatibility.
    """
    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 1024/8 = 128

        # Separate Q, K, V projections (like Qwen)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        B, S, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (critical for language modeling!)
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled Dot-Product Attention (Flash Attention backend)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal
        )

        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        return self.o_proj(attn_out)


class TRMBlock(nn.Module):
    """
    Transformer Block with RoPE (Qwen-compatible structure).

    Structure (same order as Qwen):
        1. RMSNorm -> RoPE Attention -> Residual
        2. RMSNorm -> SwiGLU FFN -> Residual
    """
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 8,
        expansion: int = 4,
        eps: float = 1e-6
    ):
        super().__init__()
        d_ff = expansion * d_model

        # Attention sublayer
        self.norm1 = nn.RMSNorm(d_model, eps=eps)
        self.attn = TRMAttention(d_model, num_heads)

        # FFN sublayer (SwiGLU)
        self.norm2 = nn.RMSNorm(d_model, eps=eps)
        self.mlp = SwiGLU(d_model, d_ff, d_model)

    def forward(
        self,
        h: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            h: Pre-fused input (x+y+z or y+z) [B, S, D]
            cos, sin: RoPE embeddings [1, 1, S, head_dim]
        Returns:
            output: Processed state [B, S, D]
        """
        # Attention with RoPE
        h = h + self.attn(self.norm1(h), cos, sin, is_causal)

        # SwiGLU FFN
        h = h + self.mlp(self.norm2(h))

        return h


# Backward compatibility alias
TRMTransformerBlock = TRMBlock
