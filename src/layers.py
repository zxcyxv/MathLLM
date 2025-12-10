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

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Only convert dtype if necessary (avoid unnecessary conversion)
        target_dtype = x.dtype if x is not None else cos.dtype
        if cos.dtype != target_dtype:
            cos = cos.to(target_dtype)
            sin = sin.to(target_dtype)

        # Return shape: [1, 1, S, head_dim] for broadcasting with [B, num_heads, S, head_dim]
        return (cos[None, None, :, :], sin[None, None, :, :])


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
    RoPE-enabled Causal Self-Attention (Qwen-compatible) with KV cache support.
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
        is_causal: bool = True,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: Input tensor [B, S, D] or [B, 1, D] for incremental decoding
            cos, sin: RoPE embeddings
            is_causal: Use causal attention mask
            past_kv: Cached (k, v) from past tokens [B, num_heads, past_len, head_dim]
            use_cache: Whether to return new KV for caching
            attention_mask: [B, S] or [B, 1, S, S] mask for padded sequences

        Returns:
            output: Attention output [B, S, D]
            new_kv: (k, v) tuple if use_cache=True
        """
        B, S, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (critical for language modeling!)
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache: concat past KV with current
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Prepare new KV for cache (before attention, after RoPE)
        new_kv = (k, v) if use_cache else None

        # Build attention mask for SDPA
        # When using KV cache, is_causal should be False (we handle causality via cache structure)
        use_causal = is_causal and past_kv is None
        attn_mask = None

        if attention_mask is not None and not use_causal:
            # Convert [B, total_len] to [B, 1, 1, total_len] for broadcasting
            # total_len = past_len + current_len
            total_len = k.size(2)
            if attention_mask.dim() == 2:
                # Expand to [B, 1, 1, total_len]
                attn_mask = attention_mask[:, None, None, :total_len]
                # Convert to additive mask: 0 -> 0, 1 -> -inf (but we want opposite)
                # attention_mask: 1 = attend, 0 = ignore
                attn_mask = attn_mask.to(dtype=q.dtype)
                attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min

        # Scaled Dot-Product Attention (Flash Attention backend)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=use_causal
        )

        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        output = self.o_proj(attn_out)

        if use_cache:
            return output, new_kv
        return output


class TRMBlock(nn.Module):
    """
    Transformer Block with RoPE (Qwen-compatible structure) and KV cache support.

    Structure (same order as Qwen):
        1. RMSNorm -> RoPE Attention -> Residual
        2. RMSNorm -> SwiGLU FFN -> Residual

    Zero Init: Output projections (o_proj, down_proj) are initialized to zero
    so that initial TRM output is identity (h + 0 = h), preserving Qwen's
    performance at initialization and enabling stable early training.
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

        # Zero Init for stable training start
        # Makes initial block output = 0, so residual h + 0 = h (identity)
        self._zero_init_output_projections()

    def _zero_init_output_projections(self):
        """Zero-initialize output projections for identity behavior at init."""
        nn.init.zeros_(self.attn.o_proj.weight)
        nn.init.zeros_(self.mlp.down_proj.weight)

    def forward(
        self,
        h: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            h: Pre-fused input (x+y+z or y+z) [B, S, D] or [B, 1, D]
            cos, sin: RoPE embeddings [1, 1, S, head_dim]
            past_kv: Cached KV from past tokens
            use_cache: Whether to return KV cache
            attention_mask: [B, total_len] mask for padded sequences
        Returns:
            output: Processed state [B, S, D]
            new_kv: (k, v) tuple if use_cache=True
        """
        # TRM uses DIRECT REPLACEMENT (no residual!)
        # This is different from standard Transformers
        # With Zero Init, output starts as 0, not as input

        # Attention with RoPE (no residual)
        if use_cache:
            h_attn, new_kv = self.attn(
                self.norm1(h), cos, sin, is_causal, past_kv,
                use_cache=True, attention_mask=attention_mask
            )
        else:
            h_attn = self.attn(
                self.norm1(h), cos, sin, is_causal, past_kv,
                use_cache=False, attention_mask=attention_mask
            )
            new_kv = None

        # SwiGLU FFN (no residual)
        out = self.mlp(self.norm2(h_attn))

        if use_cache:
            return out, new_kv
        return out


# Backward compatibility alias
TRMTransformerBlock = TRMBlock
