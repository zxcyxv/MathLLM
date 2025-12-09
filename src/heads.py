"""
Part 3: TRM Output Head
LM Head with Final Norm and Qwen weight initialization
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import TRMConfig


class TRMHeads(nn.Module):
    """
    Output head for TRM with Qwen-compatible structure.

    Features:
        1. Final RMSNorm: Stabilizes output distribution to match Qwen's expectations
        2. LM Head: Projects y state to vocabulary logits
        3. Smart Init: Can initialize from Qwen's lm_head via SVD compression
    """

    def __init__(self, config: TRMConfig, qwen_lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # Final Norm (critical: matches Qwen's output distribution)
        self.norm = nn.RMSNorm(config.d_lat, eps=config.eps)

        # LM Head: d_lat -> vocab_size
        self.lm_head = nn.Linear(config.d_lat, config.vocab_size, bias=False)

        # Initialize weights
        if qwen_lm_head is not None:
            self._init_from_qwen(qwen_lm_head)
        else:
            self._init_weights()

    def _init_weights(self):
        """Default Xavier initialization"""
        nn.init.xavier_uniform_(self.lm_head.weight)

    def _init_from_qwen(self, qwen_lm_head: nn.Module):
        """
        Initialize TRM lm_head from Qwen's pretrained weights via SVD compression.

        Qwen lm_head: [vocab_size, 3584]
        TRM lm_head:  [vocab_size, 1024]

        SVD approach: W_qwen ≈ U @ S @ V^T
        We take the top-1024 components to preserve the most important directions.
        """
        print("[TRMHeads] Initializing from Qwen lm_head via SVD...")

        with torch.no_grad():
            # Get Qwen weights: [vocab_size, 3584]
            W_qwen = qwen_lm_head.weight.float()
            qwen_vocab_size = W_qwen.shape[0]
            trm_vocab_size = self.config.vocab_size

            # Handle vocab size mismatch (Qwen may have different vocab size)
            if qwen_vocab_size != trm_vocab_size:
                print(f"[TRMHeads] Vocab size mismatch: Qwen={qwen_vocab_size}, TRM config={trm_vocab_size}")
                # Use the smaller vocab size
                min_vocab = min(qwen_vocab_size, trm_vocab_size)
                W_qwen = W_qwen[:min_vocab, :]

            # SVD: W = U @ diag(S) @ V^T
            # U: [vocab_size, vocab_size], S: [min(V,3584)], V: [3584, 3584]
            # For efficiency, use lowrank SVD
            try:
                U, S, V = torch.svd_lowrank(W_qwen, q=self.config.d_lat)
                # U: [vocab_size, 1024], S: [1024], V: [3584, 1024]

                # Reconstruct compressed weights: W_trm = U @ diag(S)
                # This gives us [vocab_size, 1024]
                W_trm = U @ torch.diag(S)

                # If TRM vocab is larger, pad with zeros
                if trm_vocab_size > W_trm.shape[0]:
                    padding = torch.zeros(trm_vocab_size - W_trm.shape[0], W_trm.shape[1],
                                         dtype=W_trm.dtype, device=W_trm.device)
                    W_trm = torch.cat([W_trm, padding], dim=0)

                self.lm_head.weight.copy_(W_trm.to(self.lm_head.weight.dtype))
                print(f"[TRMHeads] SVD compression successful: {W_qwen.shape} -> {W_trm.shape}")
                print(f"[TRMHeads] Top singular values: {S[:5].tolist()}")

            except Exception as e:
                print(f"[TRMHeads] SVD failed ({e}), using scale-matched random init...")
                # Fallback: match the scale of Qwen weights
                scale = W_qwen.std().item()
                nn.init.normal_(self.lm_head.weight, mean=0.0, std=scale)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute token logits from solution state y.

        Args:
            y: Solution state [B, S, D]

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
