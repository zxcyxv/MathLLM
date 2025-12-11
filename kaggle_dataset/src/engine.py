"""
Part 2: TinyRecursiveTransformer Engine
Implements Level 3: Latent Recursion (Paper Figure 3)
With RoPE support for language modeling capability
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn

from .config import TRMConfig
from .layers import TRMBlock


class TinyRecursiveTransformer(nn.Module):
    """
    Level 3: Latent Recursion (Paper Figure 3)

    One latent_recursion call:
        - n times: z = net(x + y + z)   # Update reasoning state
        - 1 time:  y = net(y + z)       # Update solution state (x excluded!)

    With RoPE support for proper sequence modeling.
    Supports KV cache for efficient inference.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.n = config.n_latent  # n = 6
        self.num_virtual_layers = self.n + 1  # n for z updates + 1 for y update

        # Single shared transformer block with RoPE
        self.block = TRMBlock(
            d_model=config.d_lat,
            num_heads=config.num_heads,
            expansion=config.expansion,
            eps=config.eps
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Xavier initialization for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        One latent_recursion call (Paper Figure 3, Page 5).

        Args:
            x: Context state [B, S, D] - anchor from backbone
            y: Solution state [B, S, D] - current answer embedding
            z: Reasoning state [B, S, D] - hidden reasoning path
            cos, sin: RoPE embeddings [1, 1, S, head_dim]
            past_kvs: List of (k, v) tuples for each virtual layer (n+1 total)
                      Each (k, v) has shape [B, num_heads, past_len, head_dim]
            use_cache: Whether to return updated KV cache
            attention_mask: [B, total_len] mask for padded sequences (batch > 1)

        Returns:
            y_new: Updated solution state [B, S, D]
            z_new: Updated reasoning state [B, S, D]
            new_kvs: Updated KV cache list if use_cache=True, else None
        """
        # Direct replacement (no residual) per TRM paper:
        #   z = net(x + y + z)
        #   y = net(y + z)

        new_kvs = [] if use_cache else None

        # n times z update (Reasoning Mode)
        for i in range(self.n):
            h = x + y + z                  # Additive fusion
            past_kv_i = past_kvs[i] if past_kvs else None

            if use_cache:
                z, kv_i = self.block(
                    h, cos, sin, past_kv=past_kv_i,
                    use_cache=True, attention_mask=attention_mask
                )
                new_kvs.append(kv_i)
            else:
                z = self.block(
                    h, cos, sin, past_kv=past_kv_i,
                    attention_mask=attention_mask
                )

        # 1 time y update (Prediction Mode) - x is completely excluded, not masked!
        h = y + z                          # x excluded
        past_kv_y = past_kvs[self.n] if past_kvs else None

        if use_cache:
            y, kv_y = self.block(
                h, cos, sin, past_kv=past_kv_y,
                use_cache=True, attention_mask=attention_mask
            )
            new_kvs.append(kv_y)
        else:
            y = self.block(
                h, cos, sin, past_kv=past_kv_y,
                attention_mask=attention_mask
            )

        return y, z, new_kvs
