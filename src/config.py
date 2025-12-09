"""
TRM Configuration
Hyperparameters for Qwen-TRM Integrated Model
"""

from dataclasses import dataclass


@dataclass
class TRMConfig:
    """Configuration for TRM (Tiny Recursive Model)"""

    # Backbone dimensions
    backbone_dim: int = 3584      # Qwen-2.5-Math-7B hidden size
    d_lat: int = 1024             # TRM latent dimension
    num_heads: int = 8            # Attention heads (1024/8 = 128 head_dim, same as Qwen)
    expansion: int = 4            # FFN expansion ratio

    # RoPE settings (same as Qwen for compatibility)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # 3-Level Loop Parameters (Paper Figure 3)
    n_latent: int = 6             # Level 3: Latent Recursion (z updates)
    T_recursion: int = 3          # Level 2: Deep Recursion (T-1 no_grad + 1 grad)
    N_supervision: int = 16       # Level 1: Deep Supervision (backward each step)

    # Vocabulary
    vocab_size: int = 151936      # Qwen tokenizer vocab size

    # Normalization
    eps: float = 1e-6

    # EMA (Paper recommendation)
    use_ema: bool = True
    ema_decay: float = 0.999

    @property
    def d_ff(self) -> int:
        """FFN hidden dimension"""
        return self.expansion * self.d_lat

    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.d_lat // self.num_heads
