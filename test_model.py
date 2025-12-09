"""
Quick test to verify model shapes and forward pass
Run this locally before Kaggle to catch errors early
"""

import torch


def test_trm_components():
    """Test TRM components without loading Qwen backbone"""
    print("=" * 50)
    print("Testing TRM Components (with RoPE)")
    print("=" * 50)

    from src.config import TRMConfig
    from src.interface import TRMInterface
    from src.engine import TinyRecursiveTransformer
    from src.heads import TRMHeads
    from src.layers import RotaryEmbedding

    config = TRMConfig()
    print(f"\nConfig:")
    print(f"  d_lat={config.d_lat}")
    print(f"  num_heads={config.num_heads} (head_dim={config.d_lat // config.num_heads})")
    print(f"  rope_theta={config.rope_theta}")
    print(f"  n_latent={config.n_latent} (Level 3: Latent Recursion)")
    print(f"  T_recursion={config.T_recursion} (Level 2: Deep Recursion)")
    print(f"  N_supervision={config.N_supervision} (Level 1: Deep Supervision)")

    # Test dimensions
    B, S = 2, 128  # Batch, Sequence
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Test shape: [B={B}, S={S}]")

    # 1. TRMInterface
    print("\n[1] TRMInterface")
    interface = TRMInterface(config).to(device)
    fake_hidden = torch.randn(B, S, config.backbone_dim, device=device)
    x = interface.extract_context(fake_hidden)
    y, z = interface.initialize_states(x)
    print(f"  Input:  {fake_hidden.shape} (Qwen hidden)")
    print(f"  x:      {x.shape}")
    print(f"  y:      {y.shape}")
    print(f"  z:      {z.shape}")
    assert x.shape == (B, S, config.d_lat), "x shape mismatch"
    print("  ✓ Shapes correct")

    # 2. RotaryEmbedding
    print("\n[2] RotaryEmbedding")
    head_dim = config.d_lat // config.num_heads
    rotary_emb = RotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta
    ).to(device)
    cos, sin = rotary_emb(x, S)
    print(f"  head_dim: {head_dim}")
    print(f"  cos:    {cos.shape}")
    print(f"  sin:    {sin.shape}")
    # Shape: [1, 1, S, head_dim] for broadcasting with [B, num_heads, S, head_dim]
    assert cos.shape == (1, 1, S, head_dim), "cos shape mismatch"
    assert sin.shape == (1, 1, S, head_dim), "sin shape mismatch"
    print("  ✓ RoPE shapes correct")

    # 3. TinyRecursiveTransformer (Level 3: Latent Recursion)
    print("\n[3] TinyRecursiveTransformer (with RoPE)")
    engine = TinyRecursiveTransformer(config).to(device)
    y_new, z_new = engine(x, y, z, cos, sin)
    print(f"  y_new:  {y_new.shape}")
    print(f"  z_new:  {z_new.shape}")
    assert y_new.shape == (B, S, config.d_lat), "y_new shape mismatch"
    assert z_new.shape == (B, S, config.d_lat), "z_new shape mismatch"
    print("  ✓ Shapes correct")

    # 4. TRMHeads (with Final Norm)
    print("\n[4] TRMHeads (with Final Norm)")
    heads = TRMHeads(config).to(device)
    logits = heads(y_new)
    print(f"  logits: {logits.shape}")
    assert logits.shape == (B, S, config.vocab_size), "logits shape mismatch"
    print("  ✓ Shapes correct")

    # 5. Memory usage
    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n[5] GPU Memory (TRM only): {mem_mb:.1f} MB")

    print("\n" + "=" * 50)
    print("All component tests passed! ✓")
    print("=" * 50)


def test_full_forward():
    """Test full forward pass with fake backbone output"""
    print("\n" + "=" * 50)
    print("Testing Full Forward Pass (Fake Backbone)")
    print("=" * 50)

    from src.config import TRMConfig
    from src.model import QwenTRM

    config = TRMConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, S = 2, 128

    # Create model without backbone
    model = QwenTRM(config).to(device)

    # Create a fake backbone that just returns random hidden states
    class FakeBackbone(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            # Fake lm_head for SVD init test
            self.lm_head = torch.nn.Linear(hidden_size, config.vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
            B, S = input_ids.shape
            hidden = torch.randn(B, S, self.hidden_size, device=input_ids.device)
            return type('obj', (object,), {'last_hidden_state': hidden})()

    fake_backbone = FakeBackbone(config.backbone_dim).to(device)
    model.set_backbone(fake_backbone, init_lm_head=True)

    # Test forward
    input_ids = torch.randint(0, 1000, (B, S), device=device)
    labels = input_ids.clone()

    print(f"\nInput: {input_ids.shape}")

    # Test one deep_recursion call (Level 2)
    print("\n[1] Single deep_recursion call (Level 2)")
    output = model(input_ids=input_ids, labels=labels)
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  y (detached): {output['y'].shape}")
    print(f"  z (detached): {output['z'].shape}")

    # Test chaining supervision steps (Level 1 simulation)
    print("\n[2] Chained supervision steps (Level 1 simulation, 3 steps)")
    y, z = None, None
    for step in range(3):
        output = model(input_ids=input_ids, labels=labels, y=y, z=z)
        y, z = output['y'], output['z']
        print(f"  Step {step+1}: Loss={output['loss'].item():.4f}")

    # Inference forward
    print("\n[3] Inference forward (no labels)")
    output_inf = model(input_ids=input_ids)
    print(f"  Inference logits: {output_inf['logits'].shape}")

    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n[4] GPU Memory (Full forward): {mem_mb:.1f} MB")

    print("\n" + "=" * 50)
    print("Full forward pass test passed! ✓")
    print("=" * 50)


def test_parameter_count():
    """Count trainable parameters"""
    print("\n" + "=" * 50)
    print("Parameter Count")
    print("=" * 50)

    from src.config import TRMConfig
    from src.interface import TRMInterface
    from src.engine import TinyRecursiveTransformer
    from src.heads import TRMHeads

    config = TRMConfig()

    interface = TRMInterface(config)
    engine = TinyRecursiveTransformer(config)
    heads = TRMHeads(config)

    def count_params(module, name):
        total = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {total / 1e6:.2f}M")
        return total

    print("\nTrainable TRM parameters:")
    p1 = count_params(interface, "TRMInterface")
    p2 = count_params(engine, "TinyRecursiveTransformer")
    p3 = count_params(heads, "TRMHeads")

    total = p1 + p2 + p3
    print(f"\n  Total TRM: {total / 1e6:.2f}M")
    print(f"  (Qwen backbone: ~7000M, frozen)")

    # Effective depth calculation
    n_layers = 2  # Paper: 2-layer block
    n = config.n_latent  # 6
    T = config.T_recursion  # 3
    N_sup = config.N_supervision  # 16

    depth_per_sup = n_layers * (n + 1) * T
    total_depth = depth_per_sup * N_sup

    print(f"\n  Effective Depth:")
    print(f"    Per supervision step: {n_layers} × ({n}+1) × {T} = {depth_per_sup}")
    print(f"    Total: {depth_per_sup} × {N_sup} = {total_depth} layers")

    # Architecture summary
    print(f"\n  Architecture:")
    print(f"    head_dim: {config.d_lat // config.num_heads} (same as Qwen)")
    print(f"    RoPE theta: {config.rope_theta}")


if __name__ == "__main__":
    test_trm_components()
    test_full_forward()
    test_parameter_count()
