"""
Part 1: TRM Interface
Now simplified: Qwen hidden_states → TRM directly (same dimension)
No projection needed since d_lat = backbone_dim = 3584
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

from .config import TRMConfig


class TRMInterface(nn.Module):
    """
    Simplified interface: No projection needed.
    TRM now operates at the same dimension as Qwen (3584).
    Only handles state initialization.
    """

    def __init__(self, config: Optional[TRMConfig] = None, **kwargs):
        super().__init__()

        # Support both config object and direct kwargs
        if config is not None:
            trm_dim = config.d_lat
        else:
            trm_dim = kwargs.get('trm_dim', 3584)

        self.trm_dim = trm_dim

        # Learnable initial states
        self.y_init = nn.Parameter(torch.zeros(1, 1, trm_dim))

        # DIS: Time step embedding (optional, only if use_dis=True)
        if config is not None and config.use_dis:
            self.time_embedding = nn.Embedding(config.dis_N_supervision, trm_dim)
            nn.init.normal_(self.time_embedding.weight, std=0.02)
        else:
            self.time_embedding = None

    def extract_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Identity function - no projection needed.

        Args:
            hidden_states: [Batch, Seq, 3584] from Qwen
        Returns:
            x: [Batch, Seq, 3584] context for TRM (same tensor)
        """
        return hidden_states

    def initialize_states(self, x: torch.Tensor, step_index: Optional[int] = None):
        """
        Initialize y and z based on context x.

        Args:
            x: [Batch, Seq, D] context (D=3584)
            step_index: Optional supervision step index for DIS time embedding
        Returns:
            y: [Batch, Seq, D] initial solution state (zeros)
            z: [Batch, Seq, D] initial reasoning state (zeros)

        Note: Both y and z start as zeros. With Zero Init in TRMBlock,
        this ensures stable initialization where the first forward pass
        outputs x (the context), and values don't explode during recursion.

        DIS: If step_index is provided and time_embedding exists, adds time
        step conditioning to y so the model knows which denoising step it's at.
        """
        batch_size, seq_len, d = x.shape

        # y: starts as learnable neutral state (zeros initially)
        y = self.y_init.expand(batch_size, seq_len, -1).clone()

        # DIS: Add time step embedding if available
        if self.time_embedding is not None and step_index is not None:
            step_tensor = torch.tensor([step_index], device=x.device, dtype=torch.long)
            time_emb = self.time_embedding(step_tensor)  # [1, D]
            y = y + time_emb.unsqueeze(1)  # Broadcast to [B, S, D]

        # z: starts as zeros (not x!) to prevent value explosion
        z = torch.zeros(batch_size, seq_len, d, device=x.device, dtype=x.dtype)

        return y, z
