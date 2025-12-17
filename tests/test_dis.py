"""
Unit tests for DIS (Deep Improvement Supervision) implementation
"""

import torch
from src.config import TRMConfig
from src.dis_utils import DISTargetGenerator
from src.interface import TRMInterface


def test_noise_schedule_linear():
    """Test linear noise schedule"""
    generator = DISTargetGenerator(
        vocab_size=1000,
        N_supervision=6,
        noise_schedule="linear"
    )

    # Check noise levels decrease or stay same (monotonic non-increasing)
    for i in range(5):
        assert generator.noise_levels[i] >= generator.noise_levels[i+1], \
            f"Noise should not increase: step {i} = {generator.noise_levels[i]}, step {i+1} = {generator.noise_levels[i+1]}"

    # Last step should be 0 (no noise)
    assert generator.noise_levels[-1] == 0.0, "Last step should have zero noise"

    # First step should be capped at 0.5
    assert generator.noise_levels[0] <= 0.5, "First step noise should be capped at 0.5"

    # Check that some noise levels are different (not all the same)
    unique_levels = len(set(generator.noise_levels))
    assert unique_levels > 1, f"Should have varying noise levels, got {unique_levels} unique values"

    print("✓ Linear noise schedule test passed")


def test_noise_schedule_cosine():
    """Test cosine noise schedule"""
    generator = DISTargetGenerator(
        vocab_size=1000,
        N_supervision=6,
        noise_schedule="cosine"
    )

    # Check noise levels decrease or stay same (monotonic non-increasing)
    for i in range(5):
        assert generator.noise_levels[i] >= generator.noise_levels[i+1], \
            f"Noise should not increase: step {i} = {generator.noise_levels[i]}, step {i+1} = {generator.noise_levels[i+1]}"

    # Last step should be 0 (no noise)
    assert abs(generator.noise_levels[-1]) < 1e-6, "Last step should have ~zero noise"

    print("✓ Cosine noise schedule test passed")


def test_token_corruption():
    """Test token corruption preserves -100 positions"""
    generator = DISTargetGenerator(
        vocab_size=1000,
        N_supervision=6,
        noise_schedule="linear"
    )

    # Create test labels with some -100 (masked) positions
    labels = torch.tensor([
        [1, 2, 3, -100, -100, 5, 6],
        [10, -100, 12, 13, 14, -100, 16]
    ])

    # Corrupt with high noise
    corrupted = generator.corrupt_tokens(labels, beta=0.5, device='cpu')

    # Check -100 positions are preserved
    assert (corrupted == -100).sum() == (labels == -100).sum(), \
        "All -100 positions should be preserved"

    mask_positions = (labels == -100)
    assert torch.all(corrupted[mask_positions] == -100), \
        "Masked positions should remain -100"

    # With zero noise, should be identical
    no_noise = generator.corrupt_tokens(labels, beta=0.0, device='cpu')
    assert torch.all(no_noise == labels), "Zero noise should preserve all tokens"

    print("✓ Token corruption test passed")


def test_target_generation():
    """Test progressive target generation"""
    generator = DISTargetGenerator(
        vocab_size=1000,
        N_supervision=6,
        noise_schedule="linear"
    )

    labels = torch.tensor([
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14]
    ])

    targets = generator.generate_targets(labels, device='cpu')

    # Should return N_supervision targets
    assert len(targets) == 6, f"Should have 6 targets, got {len(targets)}"

    # All targets should have same shape
    for t in targets:
        assert t.shape == labels.shape, f"Target shape mismatch: {t.shape} vs {labels.shape}"

    # Last target should be identical to ground truth
    assert torch.all(targets[-1] == labels), "Last target should be ground truth"

    # Earlier targets should have more differences
    diff_0 = (targets[0] != labels).sum().item()
    diff_3 = (targets[3] != labels).sum().item()
    assert diff_0 >= diff_3, "Earlier targets should have more corruption"

    print("✓ Target generation test passed")


def test_time_embedding():
    """Test time step embedding in TRMInterface"""
    config = TRMConfig(use_dis=True, dis_N_supervision=6)
    interface = TRMInterface(config)

    # Check time embedding exists
    assert interface.time_embedding is not None, "Time embedding should exist in DIS mode"
    assert interface.time_embedding.num_embeddings == 6, "Should have 6 time steps"

    # Test state initialization with step index
    x = torch.randn(2, 10, config.d_lat)
    y, z = interface.initialize_states(x, step_index=0)

    assert y.shape == x.shape, f"y shape mismatch: {y.shape} vs {x.shape}"
    assert z.shape == x.shape, f"z shape mismatch: {z.shape} vs {x.shape}"

    # Different steps should produce different y initializations
    y1, _ = interface.initialize_states(x, step_index=0)
    y2, _ = interface.initialize_states(x, step_index=3)

    assert not torch.allclose(y1, y2), "Different steps should produce different y states"

    print("✓ Time embedding test passed")


def test_active_properties():
    """Test active_* properties switch correctly"""
    # DIS mode
    config_dis = TRMConfig(use_dis=True)
    assert config_dis.active_n_latent == 2, f"DIS n_latent should be 2, got {config_dis.active_n_latent}"
    assert config_dis.active_T_recursion == 1, f"DIS T_recursion should be 1, got {config_dis.active_T_recursion}"
    assert config_dis.active_N_supervision == 6, f"DIS N_supervision should be 6, got {config_dis.active_N_supervision}"

    # Standard mode
    config_std = TRMConfig(use_dis=False)
    assert config_std.active_n_latent == 6, f"Standard n_latent should be 6, got {config_std.active_n_latent}"
    assert config_std.active_T_recursion == 3, f"Standard T_recursion should be 3, got {config_std.active_T_recursion}"
    assert config_std.active_N_supervision == 16, f"Standard N_supervision should be 16, got {config_std.active_N_supervision}"

    print("✓ Active properties test passed")


def test_effective_depth_reduction():
    """Test that DIS reduces effective depth as expected"""
    config_std = TRMConfig(use_dis=False)
    config_dis = TRMConfig(use_dis=True)

    depth_std = 2 * (config_std.active_n_latent + 1) * config_std.active_T_recursion * config_std.active_N_supervision
    depth_dis = 2 * (config_dis.active_n_latent + 1) * config_dis.active_T_recursion * config_dis.active_N_supervision

    assert depth_std == 672, f"Standard depth should be 672, got {depth_std}"
    assert depth_dis == 36, f"DIS depth should be 36, got {depth_dis}"

    speedup = depth_std / depth_dis
    assert abs(speedup - 18.67) < 0.1, f"Speedup should be ~18.7x, got {speedup:.2f}x"

    print(f"✓ Effective depth reduction test passed: {speedup:.2f}x speedup")


if __name__ == "__main__":
    print("\nRunning DIS unit tests...\n")

    test_noise_schedule_linear()
    test_noise_schedule_cosine()
    test_token_corruption()
    test_target_generation()
    test_time_embedding()
    test_active_properties()
    test_effective_depth_reduction()

    print("\n✅ All tests passed!")
