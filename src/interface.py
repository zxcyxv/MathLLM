"""
Part 1: TRM Interface
Qwen (3584) → Projection (1024) → TRM States (x, y, z)
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

from .config import TRMConfig


class TRMInterface(nn.Module):
    """
    Bridge between frozen Qwen backbone and trainable TRM engine.
    Handles dimensionality reduction (Information Bottleneck).
    """

    def __init__(self, config: Optional[TRMConfig] = None, **kwargs):
        super().__init__()

        # Support both config object and direct kwargs
        if config is not None:
            backbone_dim = config.backbone_dim
            trm_dim = config.d_lat
            eps = config.eps
        else:
            backbone_dim = kwargs.get('backbone_dim', 3584)
            trm_dim = kwargs.get('trm_dim', 1024)
            eps = kwargs.get('eps', 1e-6)

        self.backbone_dim = backbone_dim
        self.trm_dim = trm_dim

        # Projection Layer (The Bottleneck)
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, trm_dim, bias=False),
            nn.RMSNorm(trm_dim, eps=eps)
        )

        # Learnable initial states
        self.y_init = nn.Parameter(torch.zeros(1, 1, trm_dim))

    def extract_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project Qwen hidden states to TRM latent space.

        Args:
            hidden_states: [Batch, Seq, 3584] from Qwen
        Returns:
            x: [Batch, Seq, 1024] context for TRM
        """
        # Ensure dtype consistency (backbone may output bfloat16)
        return self.projector(hidden_states.to(self.projector[0].weight.dtype))

    def initialize_states(self, x: torch.Tensor):
        """
        Initialize y and z based on context x.

        Args:
            x: [Batch, Seq, D] projected context
        Returns:
            y: [Batch, Seq, D] initial solution state (zeros)
            z: [Batch, Seq, D] initial reasoning state (zeros)

        Note: Both y and z start as zeros. With Zero Init in TRMBlock,
        this ensures stable initialization where the first forward pass
        outputs x (the context), and values don't explode during recursion.
        """
        batch_size, seq_len, d = x.shape

        # y: starts as learnable neutral state (zeros initially)
        y = self.y_init.expand(batch_size, seq_len, -1).clone()

        # z: starts as zeros (not x!) to prevent value explosion
        # With Zero Init, first iteration: h = x + 0 + 0 = x, z = block(x) ≈ x
        z = torch.zeros(batch_size, seq_len, d, device=x.device, dtype=x.dtype)

        return y, z

    def _init_weights(self, module: nn.Module):
        """Xavier initialization for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
