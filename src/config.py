"""
TRM Configuration
Hyperparameters for Qwen-TRM Integrated Model
"""

from dataclasses import dataclass


@dataclass
class TRMConfig:
    """Configuration for TRM (Tiny Recursive Model)"""

    # Backbone dimensions - TRM now matches Qwen exactly
    backbone_dim: int = 3584      # Qwen-2.5-Math-7B hidden size
    d_lat: int = 3584             # TRM latent dimension (same as Qwen!)
    num_heads: int = 28           # Attention heads (3584/28 = 128 head_dim, same as Qwen)
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

    # DIS (Deep Improvement Supervision) Mode
    use_dis: bool = False             # Enable DIS training
    dis_n_latent: int = 2             # Reduced from 6
    dis_T_recursion: int = 1          # Reduced from 3
    dis_N_supervision: int = 6        # Reduced from 16

    # Target corruption settings
    dis_noise_schedule: str = "linear"  # "linear" or "cosine"
    dis_corruption_strategy: str = "random_token"

    @property
    def active_n_latent(self) -> int:
        """Get active n_latent based on DIS mode"""
        return self.dis_n_latent if self.use_dis else self.n_latent

    @property
    def active_T_recursion(self) -> int:
        """Get active T_recursion based on DIS mode"""
        return self.dis_T_recursion if self.use_dis else self.T_recursion

    @property
    def active_N_supervision(self) -> int:
        """Get active N_supervision based on DIS mode"""
        return self.dis_N_supervision if self.use_dis else self.N_supervision

    @property
    def d_ff(self) -> int:
        """FFN hidden dimension"""
        return self.expansion * self.d_lat

    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.d_lat // self.num_heads
