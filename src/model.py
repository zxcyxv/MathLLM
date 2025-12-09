"""
QwenTRM: Full Integrated Model
Implements Level 2: Deep Recursion (Paper Figure 3)
With RoPE support and Qwen lm_head initialization
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .config import TRMConfig
from .interface import TRMInterface
from .engine import TinyRecursiveTransformer
from .heads import TRMHeads
from .layers import RotaryEmbedding


class QwenTRM(nn.Module):
    """
    Qwen-TRM Integrated Model.
    Implements Level 2: Deep Recursion (T-1 no_grad + 1 grad)

    Architecture:
        1. Qwen backbone (frozen) - Semantic encoding
        2. TRMInterface - Dimensionality reduction (3584 -> 1024)
        3. TinyRecursiveTransformer - Recursive reasoning with RoPE
        4. TRMHeads - Token prediction (initialized from Qwen)

    Features:
        - RoPE with same settings as Qwen (theta=1e6, head_dim=128)
        - LM Head initialized via SVD compression from Qwen
    """

    def __init__(
        self,
        config: TRMConfig,
        backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config
        self.T = config.T_recursion  # T = 3

        # Backbone (Qwen) - to be set later or passed in
        self.backbone = backbone

        # TRM Components
        self.interface = TRMInterface(config)
        self.engine = TinyRecursiveTransformer(config)
        self.heads = TRMHeads(config)  # Will init from Qwen later

        # RoPE (same settings as Qwen for compatibility)
        head_dim = config.d_lat // config.num_heads  # 1024/8 = 128
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def set_backbone(self, backbone: nn.Module, init_lm_head: bool = True):
        """
        Set the Qwen backbone model.

        Args:
            backbone: Qwen model (e.g., Qwen2ForCausalLM or Qwen2Model)
            init_lm_head: Whether to initialize TRM lm_head from Qwen's lm_head
        """
        self.backbone = backbone

        # Initialize TRM lm_head from Qwen's lm_head via SVD
        if init_lm_head:
            qwen_lm_head = self._get_qwen_lm_head(backbone)
            if qwen_lm_head is not None:
                self.heads.set_from_qwen(qwen_lm_head)

    def _get_qwen_lm_head(self, backbone: nn.Module) -> Optional[nn.Module]:
        """Extract lm_head from Qwen model (handles different model types)"""
        # Try different attribute names
        if hasattr(backbone, 'lm_head'):
            return backbone.lm_head
        elif hasattr(backbone, 'embed_out'):
            return backbone.embed_out
        else:
            print("[QwenTRM] Warning: Could not find lm_head in backbone")
            return None

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        One deep_recursion call (Paper Figure 3).

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Attention mask [B, S]
            labels: Target labels [B, S] (for training)
            y: Solution state from previous step [B, S, D] (None for first step)
            z: Reasoning state from previous step [B, S, D] (None for first step)

        Returns:
            dict containing:
                - loss: CE loss (if labels provided)
                - logits: Token logits [B, S, V]
                - y: Detached solution state for next step
                - z: Detached reasoning state for next step
        """
        assert self.backbone is not None, "Backbone not set. Call set_backbone() first."

        B, S = input_ids.shape

        # 1. Encode with Qwen backbone (frozen)
        with torch.no_grad():
            backbone_output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Handle both CausalLMOutput and BaseModelOutput
            if hasattr(backbone_output, 'hidden_states') and backbone_output.hidden_states is not None:
                hidden_states = backbone_output.hidden_states[-1]  # Last layer
            else:
                # Fallback for models that return last_hidden_state directly
                hidden_states = backbone_output.last_hidden_state

        # 3. Project to TRM latent space
        x = self.interface.extract_context(hidden_states)

        # 4. Initialize states if first supervision step
        if y is None or z is None:
            y, z = self.interface.initialize_states(x)

        # 5. Get RoPE embeddings
        cos, sin = self.rotary_emb(x, S)

        # 6. Deep Recursion: T-1 times no_grad + 1 time grad
        # T-1 times: "think" without gradient (memory efficient)
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.engine(x, y, z, cos, sin)

        # Last 1 time: with gradient (for learning)
        y, z = self.engine(x, y, z, cos, sin)

        # 7. Compute logits
        logits = self.heads(y)

        # 8. Build output
        output = {
            'logits': logits,
            'y': y.detach(),  # Detach for next supervision step
            'z': z.detach(),
        }

        # 9. Compute loss if training
        if labels is not None:
            output['loss'] = self._compute_loss(logits, labels)

        return output

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CE loss for LM (next token prediction).

        Args:
            logits: Token logits [B, S, V]
            labels: Target labels [B, S]

        Returns:
            Cross-entropy loss
        """
        # Shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        num_supervision_steps: int = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Attention mask [B, S]
            max_new_tokens: Maximum tokens to generate
            num_supervision_steps: Number of supervision steps (default: N_supervision)

        Returns:
            Generated token IDs [B, S + max_new_tokens]
        """
        if num_supervision_steps is None:
            num_supervision_steps = self.config.N_supervision

        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Run N_sup supervision steps for each token
                y, z = None, None
                for _ in range(num_supervision_steps):
                    output = self.forward(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        y=y, z=z
                    )
                    y = output['y']
                    z = output['z']

                # Get next token
                next_token_logits = output['logits'][:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=-1)

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(next_token)
                    ], dim=-1)

        return generated

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_name: str = "Qwen/Qwen2.5-Math-7B",
        config: Optional[TRMConfig] = None,
        device: str = "cuda",
        init_lm_head: bool = True
    ) -> "QwenTRM":
        """
        Create QwenTRM with pretrained Qwen backbone.

        Args:
            backbone_name: HuggingFace model name
            config: TRM configuration
            device: Device to load model on
            init_lm_head: Whether to initialize TRM lm_head from Qwen's

        Returns:
            QwenTRM instance with loaded backbone
        """
        from transformers import AutoModelForCausalLM

        if config is None:
            config = TRMConfig()

        # Load Qwen backbone (full model for lm_head access)
        print(f"[QwenTRM] Loading backbone: {backbone_name}")
        backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)

        # Create model
        model = cls(config=config).to(device)
        model.set_backbone(backbone, init_lm_head=init_lm_head)
        model.freeze_backbone()

        return model
