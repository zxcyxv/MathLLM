"""
Training script for QwenTRM on GSM8K
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import TRMConfig
from src.model import QwenTRM
from src.train import Trainer, TrainingConfig
from src.dataset import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train QwenTRM on GSM8K")

    # Model
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-Math-7B",
                        help="Backbone model name")

    # Data
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--train_subset", type=int, default=None,
                        help="Limit training data size (for testing)")
    parser.add_argument("--test_subset", type=int, default=None,
                        help="Limit test data size")

    # Training
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")

    # TRM
    parser.add_argument("--d_lat", type=int, default=1024,
                        help="TRM latent dimension")
    parser.add_argument("--n_recursion", type=int, default=6,
                        help="Inner loop iterations")
    parser.add_argument("--t_supervision", type=int, default=3,
                        help="Deep supervision steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("QwenTRM Training on GSM8K")
    print("=" * 50)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders
    print("\nLoading GSM8K dataset...")
    train_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_subset=args.train_subset,
        test_subset=args.test_subset
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # TRM Config
    trm_config = TRMConfig(
        backbone_dim=3584,  # Qwen2.5-Math-7B
        d_lat=args.d_lat,
        num_heads=16,
        expansion=4,
        n_recursion=args.n_recursion,
        t_supervision=args.t_supervision,
        vocab_size=tokenizer.vocab_size
    )
    print(f"\nTRM Config:")
    print(f"  d_lat: {trm_config.d_lat}")
    print(f"  n_recursion: {trm_config.n_recursion}")
    print(f"  t_supervision: {trm_config.t_supervision}")

    # Load backbone
    print(f"\nLoading backbone: {args.backbone}")
    backbone = AutoModel.from_pretrained(
        args.backbone,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Create model
    print("\nCreating QwenTRM model...")
    model = QwenTRM(config=trm_config, backbone=backbone)
    model.freeze_backbone()
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Training config
    train_config = TrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )

    # Trainer
    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        config=train_config
    )

    trainer.train()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
