"""
QwenTRM: Full Integrated Model
Implements Level 2: Deep Recursion (Paper Figure 3)
With RoPE support and Qwen lm_head initialization
"""

from typing import Optional, Dict, Any, List, Tuple

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
        2. TRMInterface - State initialization (no projection, same dimension)
        3. TinyRecursiveTransformer - Recursive reasoning with RoPE
        4. TRMHeads - Token prediction (Qwen lm_head weights copied directly)

    Features:
        - TRM operates at same dimension as Qwen (d_lat = 3584)
        - RoPE with same settings as Qwen (theta=1e6, head_dim=128, num_heads=28)
        - LM Head directly uses Qwen's pretrained weights (no SVD needed)
    """

    def __init__(
        self,
        config: TRMConfig,
        backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config
        self.T = config.active_T_recursion  # T = 3 (or 1 in DIS mode)

        # Backbone (Qwen) - to be set later or passed in
        self.backbone = backbone

        # TRM Components
        self.interface = TRMInterface(config)
        self.engine = TinyRecursiveTransformer(config)
        self.heads = TRMHeads(config)  # Will copy from Qwen later

        # RoPE (same settings as Qwen for compatibility)
        head_dim = config.d_lat // config.num_heads  # 3584/28 = 128
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
            init_lm_head: Whether to copy Qwen's lm_head weights to TRM
        """
        self.backbone = backbone

        # Copy Qwen's lm_head weights directly (same dimension, no SVD needed)
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

    def encode_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False
    ):
        """
        Encode input with backbone only (frozen, run once per batch).

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Attention mask [B, S]
            past_key_values: Cached KV from previous calls
            use_cache: Whether to return KV cache

        Returns:
            hidden_states: Backbone output [B, S, backbone_dim]
            past_key_values: (optional) KV cache if use_cache=True
        """
        assert self.backbone is not None, "Backbone not set. Call set_backbone() first."

        backbone_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        if hasattr(backbone_output, 'hidden_states') and backbone_output.hidden_states is not None:
            hidden_states = backbone_output.hidden_states[-1]
        else:
            hidden_states = backbone_output.last_hidden_state

        if use_cache:
            return hidden_states, backbone_output.past_key_values
        return hidden_states

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full encode: backbone + interface projection.

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Attention mask [B, S]

        Returns:
            x: Context tensor [B, S, d_lat]
        """
        with torch.no_grad():
            hidden_states = self.encode_backbone(input_ids, attention_mask)
        x = self.interface.extract_context(hidden_states)
        return x

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        One deep_recursion call (Paper Figure 3).

        Args:
            input_ids: Input token IDs [B, S] (optional if hidden_states provided)
            attention_mask: Attention mask [B, S]
            labels: Target labels [B, S] (for training)
            y: Solution state from previous step [B, S, D] (None for first step)
            z: Reasoning state from previous step [B, S, D] (None for first step)
            hidden_states: Pre-computed backbone output [B, S, backbone_dim]
                          (optional, skips backbone if provided)
            cos, sin: Pre-computed RoPE embeddings (optional, computed if not provided)
            step_index: Optional supervision step index for DIS time embedding

        Returns:
            dict containing:
                - loss: CE loss (if labels provided)
                - logits: Token logits [B, S, V]
                - y: Detached solution state for next step
                - z: Detached reasoning state for next step
                - cos, sin: RoPE embeddings for reuse
        """
        # Get hidden_states from backbone or use pre-computed
        if hidden_states is None:
            assert input_ids is not None, "Either input_ids or hidden_states must be provided"
            with torch.no_grad():
                hidden_states = self.encode_backbone(input_ids, attention_mask)

        # TRM operates at same dimension as Qwen (identity projection)
        # Inlined from interface.extract_context() for minimal overhead
        x = hidden_states

        B, S, _ = x.shape

        # Initialize states if first supervision step
        if y is None or z is None:
            y, z = self.interface.initialize_states(x, step_index=step_index)

        # Get RoPE embeddings (compute once, reuse across supervision steps)
        if cos is None or sin is None:
            cos, sin = self.rotary_emb(x, S)

        # Deep Recursion: T-1 times no_grad + 1 time grad
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z, _ = self.engine(x, y, z, cos, sin)

        # Last 1 time: with gradient (for learning)
        y, z, _ = self.engine(x, y, z, cos, sin)

        # Compute logits
        logits = self.heads(y)

        # Build output
        output = {
            'logits': logits,
            'y': y.detach(),
            'z': z.detach(),
            'cos': cos,
            'sin': sin,
        }

        # Compute loss if training
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
        num_supervision_steps: int = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with efficient KV cache.

        Key optimization: Only process the CURRENT token through 16 supervision steps.
        Past tokens' KV are cached and reused.

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Attention mask [B, S]
            max_new_tokens: Maximum tokens to generate
            num_supervision_steps: Number of supervision steps (default: N_supervision)
            eos_token_id: Token ID to stop generation

        Returns:
            Generated token IDs [B, S + generated_tokens]
        """
        if num_supervision_steps is None:
            num_supervision_steps = self.config.active_N_supervision

        self.eval()
        B = input_ids.size(0)
        D = self.config.d_lat
        device = input_ids.device
        generated = input_ids.clone()

        with torch.no_grad():
            # ================================================================
            # PREFILL: Process entire prompt
            # ================================================================
            hidden_states, backbone_kv = self.encode_backbone(
                input_ids, attention_mask, use_cache=True
            )
            S = hidden_states.size(1)

            # Cache x (backbone hidden states)
            x_cache = hidden_states  # [B, S, D]

            # Initialize y, z for all positions
            # DIS: Use final denoising step (least noise) for best quality
            step_index = num_supervision_steps - 1 if self.config.use_dis else None
            y_cache, z_cache = self.interface.initialize_states(x_cache, step_index=step_index)

            # Compute full RoPE for max sequence length (cache for reuse)
            max_seq = S + max_new_tokens
            cos_full, sin_full = self.rotary_emb(hidden_states, max_seq)

            # Get RoPE for prefill positions
            cos_prefill = cos_full[:, :, :S, :]
            sin_prefill = sin_full[:, :, :S, :]

            # Run 16 * T supervision for prefill, cache KV from LAST iteration
            trm_kv_cache = None
            for sup_step in range(num_supervision_steps):
                for t in range(self.T):
                    # Only cache on the very last iteration
                    is_last = (sup_step == num_supervision_steps - 1) and (t == self.T - 1)
                    y_cache, z_cache, new_kvs = self.engine(
                        x_cache, y_cache, z_cache,
                        cos_prefill, sin_prefill,
                        past_kvs=None,  # No past for prefill
                        use_cache=is_last,
                        attention_mask=attention_mask  # Pass mask for batch > 1
                    )
                    if is_last:
                        trm_kv_cache = new_kvs  # List of (k, v) for n+1 virtual layers

            # ================================================================
            # DECODE: Generate tokens one by one
            # ================================================================
            current_len = S

            for token_idx in range(max_new_tokens):
                # Predict next token from last position's y
                logits = self.heads(y_cache[:, -1:, :])  # [B, 1, V]
                next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)  # [B, 1]

                # Append to generated sequence FIRST (so EOS is included)
                generated = torch.cat([generated, next_token], dim=-1)

                # Check for EOS AFTER appending
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(next_token)
                    ], dim=-1)

                # Get new token's backbone hidden state
                new_hidden, backbone_kv = self.encode_backbone(
                    next_token, attention_mask,
                    past_key_values=backbone_kv,
                    use_cache=True
                )

                # Initialize new position's y, z
                new_y = self.interface.y_init.expand(B, 1, -1).clone()
                new_z = torch.zeros(B, 1, D, device=device, dtype=new_hidden.dtype)

                # Get RoPE for new position only
                cos_new = cos_full[:, :, current_len:current_len+1, :]
                sin_new = sin_full[:, :, current_len:current_len+1, :]

                # Run 16 * T supervision for NEW TOKEN ONLY
                # Q: current token (changes during loop)
                # K, V: from cache (past tokens, fixed) + current token
                for sup_step in range(num_supervision_steps):
                    for t in range(self.T):
                        is_last = (sup_step == num_supervision_steps - 1) and (t == self.T - 1)
                        new_y, new_z, new_kvs = self.engine(
                            x=new_hidden,  # [B, 1, D] - current token only
                            y=new_y,       # [B, 1, D] - current token's y
                            z=new_z,       # [B, 1, D] - current token's z
                            cos=cos_new,   # RoPE for position current_len
                            sin=sin_new,
                            past_kvs=trm_kv_cache,  # Past tokens' K/V
                            use_cache=is_last,
                            attention_mask=attention_mask  # Updated mask including new token
                        )
                        if is_last:
                            # Append current token's K/V to cache
                            trm_kv_cache = self._merge_kv_cache(trm_kv_cache, new_kvs)

                # Update state caches (append new position)
                x_cache = torch.cat([x_cache, new_hidden], dim=1)
                y_cache = torch.cat([y_cache, new_y], dim=1)
                z_cache = torch.cat([z_cache, new_z], dim=1)
                current_len += 1

        return generated

    def _merge_kv_cache(
        self,
        past_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Merge new token's KV into existing cache.

        Args:
            past_kvs: List of (k, v) for each virtual layer, k: [B, heads, past_len, head_dim]
            new_kvs: List of (k, v) for each virtual layer, k: [B, heads, past_len+1, head_dim]
                     (already includes past from attention forward)

        Returns:
            Updated cache with new token appended
        """
        # new_kvs already has past concatenated from TRMAttention.forward
        # Just return as-is
        return new_kvs

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
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

        # Sync TRM dimensions with actual backbone hidden size
        # Since TRM now operates at the same dimension as Qwen, both d_lat and backbone_dim must match
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        num_attention_heads = getattr(getattr(backbone, "config", None), "num_attention_heads", None)
        if hidden_size is not None:
            print(f"[QwenTRM] Detected backbone hidden_size={hidden_size}, "
                  f"setting TRMConfig.backbone_dim and d_lat accordingly.")
            config.backbone_dim = hidden_size
            config.d_lat = hidden_size  # TRM operates at same dimension!
        if num_attention_heads is not None:
            print(f"[QwenTRM] Detected backbone num_attention_heads={num_attention_heads}, "
                  f"setting TRMConfig.num_heads accordingly.")
            config.num_heads = num_attention_heads

        # Create model
        model = cls(config=config).to(device)
        model.set_backbone(backbone, init_lm_head=init_lm_head)
        model.freeze_backbone()

        return model
