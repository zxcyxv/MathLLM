"""
Part 3: TRM Output Head
Simplified: Uses Qwen's lm_head directly (shared weights)
Since TRM now operates at d_lat=3584 (same as Qwen), no projection needed.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import TRMConfig


class TRMHeads(nn.Module):
    """
    Simplified output head for TRM.

    Since TRM now operates at the same dimension as Qwen (3584),
    we can directly use Qwen's pretrained lm_head without SVD compression.

    Features:
        1. Final RMSNorm: Stabilizes output distribution
        2. LM Head: Directly uses Qwen's pretrained lm_head (shared or copied)
    """

    def __init__(self, config: TRMConfig, qwen_lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # Final Norm (matches Qwen's model.norm before lm_head)
        self.norm = nn.RMSNorm(config.d_lat, eps=config.eps)

        # LM Head: will be set from Qwen's lm_head
        # Placeholder - will be replaced with Qwen's weights
        self.lm_head = nn.Linear(config.d_lat, config.vocab_size, bias=False)

        # Initialize weights
        if qwen_lm_head is not None:
            self._init_from_qwen(qwen_lm_head)
        else:
            self._init_weights()

    def _init_weights(self):
        """Default Xavier initialization (fallback)"""
        nn.init.xavier_uniform_(self.lm_head.weight)

    def _init_from_qwen(self, qwen_lm_head: nn.Module):
        """
        Copy Qwen's lm_head weights directly (no SVD needed).

        Since d_lat = backbone_dim = 3584, dimensions match exactly.
        """
        print("[TRMHeads] Copying Qwen lm_head weights directly (same dimension)...")

        with torch.no_grad():
            W_qwen = qwen_lm_head.weight  # [vocab_size, 3584]
            qwen_vocab_size = W_qwen.shape[0]
            qwen_hidden_size = W_qwen.shape[1]

            print(f"[TRMHeads] Qwen lm_head shape: {W_qwen.shape}")
            print(f"[TRMHeads] TRM lm_head shape: {self.lm_head.weight.shape}")

            # Verify dimensions match
            assert qwen_hidden_size == self.config.d_lat, \
                f"Dimension mismatch: Qwen={qwen_hidden_size}, TRM={self.config.d_lat}"

            # Handle vocab size mismatch if any
            if qwen_vocab_size != self.config.vocab_size:
                print(f"[TRMHeads] Vocab size mismatch: Qwen={qwen_vocab_size}, TRM={self.config.vocab_size}")
                min_vocab = min(qwen_vocab_size, self.config.vocab_size)
                self.lm_head.weight[:min_vocab, :].copy_(W_qwen[:min_vocab, :])
            else:
                self.lm_head.weight.copy_(W_qwen)

            print("[TRMHeads] Qwen lm_head weights copied successfully!")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute token logits from solution state y.

        Args:
            y: Solution state [B, S, D] where D=3584

        Returns:
            logits: Token probabilities [B, S, V]
        """
        # Final norm (critical for output stability)
        y = self.norm(y)
        return self.lm_head(y)

    def set_from_qwen(self, qwen_lm_head: nn.Module):
        """
        Set lm_head weights from Qwen after initialization.
        Useful when backbone is loaded separately.
        """
        self._init_from_qwen(qwen_lm_head)
