"""
DIS (Deep Improvement Supervision) Utilities
Implements progressive target generation via discrete diffusion
"""

from typing import List
import math

import torch


class DISTargetGenerator:
    """
    Generates progressive targets for DIS training.

    Key Idea (from paper):
        Instead of supervising all steps with the final answer,
        create N_supervision intermediate targets with decreasing noise.

        Step 0: Highest noise (hardest, ~50% corrupted)
        Step N-1: No noise (easiest, ground truth)

    This provides explicit improvement signal: each step learns to
    denoise the previous step's output.
    """

    def __init__(
        self,
        vocab_size: int,
        N_supervision: int = 6,
        noise_schedule: str = "linear",
        corruption_strategy: str = "random_token"
    ):
        """
        Args:
            vocab_size: Vocabulary size for random token sampling
            N_supervision: Number of supervision steps (default 6)
            noise_schedule: "linear" or "cosine"
            corruption_strategy: "random_token" (Qwen has no mask token)
        """
        self.vocab_size = vocab_size
        self.N_sup = N_supervision
        self.noise_schedule = noise_schedule
        self.corruption_strategy = corruption_strategy

        # Pre-compute noise levels for all steps
        self.noise_levels = [self.get_noise_level(step) for step in range(N_supervision)]

    def get_noise_level(self, step: int) -> float:
        """
        Compute noise level β_s for supervision step s.

        Args:
            step: Current supervision step (0 to N-1)

        Returns:
            beta: Noise level in [0, 1]
                  step=0 → β≈0.83 (hardest)
                  step=N-1 → β=0 (easiest, no noise)

        Paper formula (linear schedule):
            β_s = (N - s) / N

        Example (N=6):
            s=0: β=6/6=1.0   (but we cap at ~0.5 to keep some signal)
            s=1: β=5/6=0.83
            ...
            s=5: β=1/6=0.17
            s=6: β=0/6=0.0   (ground truth)
        """
        s = step + 1  # Convert 0-indexed to 1-indexed for formula
        N = self.N_sup

        if self.noise_schedule == "linear":
            beta = (N - s) / N
        elif self.noise_schedule == "cosine":
            # Cosine schedule: smoother decay
            beta = math.cos((s / N) * (math.pi / 2))
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

        # Cap maximum noise at 0.5 to preserve some signal
        # (Full random replacement would be too hard)
        return min(beta, 0.5)

    def corrupt_tokens(
        self,
        labels: torch.Tensor,
        beta: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Corrupt tokens via random replacement.

        Args:
            labels: [B, S] ground truth tokens
            beta: Noise level (0=no noise, 1=full noise)
            device: Target device

        Returns:
            corrupted: [B, S] tokens with some replaced randomly

        Strategy:
            - Preserve -100 positions (padding/prompt)
            - For valid tokens: replace with random token at rate beta
            - Use Qwen's full vocabulary (no mask token available)
        """
        corrupted = labels.clone()

        # Find valid positions (not -100)
        valid_mask = (labels != -100)

        # Generate random corruption mask
        # Each valid token has `beta` probability of being replaced
        corruption_mask = torch.rand_like(labels, dtype=torch.float) < beta
        corruption_mask = corruption_mask & valid_mask  # Only corrupt valid tokens

        # Generate random tokens from vocabulary
        num_corrupted = corruption_mask.sum().item()
        if num_corrupted > 0:
            random_tokens = torch.randint(
                0, self.vocab_size,
                (num_corrupted,),
                device=device,
                dtype=labels.dtype
            )
            corrupted[corruption_mask] = random_tokens

        return corrupted

    def generate_targets(
        self,
        labels: torch.Tensor,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Generate N_supervision progressive targets.

        Args:
            labels: [B, S] ground truth tokens
            device: Target device

        Returns:
            targets: List of N_sup tensors, each [B, S]
                     targets[0]: Most corrupted (hardest)
                     targets[N-1]: Ground truth (easiest)

        Example (N=6, beta values):
            targets[0]: β=0.5  → ~50% tokens random
            targets[1]: β=0.42 → ~42% tokens random
            targets[2]: β=0.33 → ~33% tokens random
            targets[3]: β=0.25 → ~25% tokens random
            targets[4]: β=0.17 → ~17% tokens random
            targets[5]: β=0.0  → ground truth (no corruption)
        """
        targets = []

        for step in range(self.N_sup):
            beta = self.noise_levels[step]

            if beta == 0.0:
                # Last step: use ground truth directly
                targets.append(labels.clone())
            else:
                # Intermediate steps: corrupt tokens
                corrupted = self.corrupt_tokens(labels, beta, device)
                targets.append(corrupted)

        return targets

    def get_step_info(self, step: int) -> dict:
        """
        Get information about a specific supervision step.

        Useful for logging and debugging.

        Args:
            step: Supervision step (0 to N-1)

        Returns:
            info: Dictionary with step details
        """
        beta = self.noise_levels[step]
        return {
            "step": step,
            "beta": beta,
            "corruption_rate": f"{beta * 100:.1f}%",
            "difficulty": "hardest" if step == 0 else ("easiest" if step == self.N_sup - 1 else "intermediate")
        }

    def __repr__(self) -> str:
        return (
            f"DISTargetGenerator(\n"
            f"  N_supervision={self.N_sup},\n"
            f"  noise_schedule='{self.noise_schedule}',\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  noise_levels={[f'{b:.2f}' for b in self.noise_levels]}\n"
            f")"
        )
