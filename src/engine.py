"""
Part 2: TinyRecursiveTransformer Engine
Implements Level 3: Latent Recursion (Paper Figure 3)
With RoPE support for language modeling capability
"""

from typing import Tuple, Optional

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
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.n = config.n_latent  # n = 6

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
        sin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One latent_recursion call (Paper Figure 3, Page 5).

        Args:
            x: Context state [B, S, D] - anchor from backbone
            y: Solution state [B, S, D] - current answer embedding
            z: Reasoning state [B, S, D] - hidden reasoning path
            cos, sin: RoPE embeddings [1, 1, S, head_dim]

        Returns:
            y_new: Updated solution state [B, S, D]
            z_new: Updated reasoning state [B, S, D]
        """
        # n times z update (Reasoning Mode)
        # z = net(x + y + z) - direct replacement, no residual
        for _ in range(self.n):
            h = x + y + z          # Additive fusion
            z = self.block(h, cos, sin)  # Direct replacement

        # 1 time y update (Prediction Mode)
        # y = net(y + z) - x is completely excluded, not masked!
        h = y + z                  # x excluded
        y = self.block(h, cos, sin)  # Direct replacement

        return y, z
