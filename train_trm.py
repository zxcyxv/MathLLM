"""
TRM Training Script for Qwen-2.5-Math-7B + TRM
Phase 2: Experimental group for LoRA comparison

Implements the 3-level recursive training from TRM paper.

Optimizations applied:
- Pre-tokenization (tokenize once at dataset creation)
- Dynamic padding (pad to max length in batch, not max_length)
- Mixed precision training (bfloat16)
- Optimized DataLoader (prefetch_factor, drop_last)
"""

import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.model import QwenTRM
from src.config import TRMConfig
from src.train import Trainer, TrainingConfig


def format_example(example: dict) -> str:
    """Format GSM8K example for training."""
    question = example["question"]
    answer = example["answer"]
    return f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution: {answer}"""


class MathDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for math problems with pre-tokenization.

    Optimization: Tokenizes all examples once at initialization,
    avoiding repeated tokenizer calls during training.
    Uses fixed-length padding for consistent tensor shapes (better GPU performance).
    """

    def __init__(self, dataset, tokenizer, max_length=1024):
        self.max_length = max_length

        # Pre-tokenize all examples with fixed-length padding
        print("Pre-tokenizing dataset...")
        self.tokenized_data = []
        for example in tqdm(dataset, desc="Tokenizing"):
            text = format_example(example)
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",  # Fixed padding for consistent shapes
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)

            # Labels: mask padding with -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.tokenized_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })
        print(f"Pre-tokenized {len(self.tokenized_data)} examples")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


def count_trainable_params(model):
    """Count trainable parameters (TRM only, not backbone)."""
    trainable = 0
    for name, p in model.named_parameters():
        if p.requires_grad and "backbone" not in name:
            trainable += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser(description="Train Qwen + TRM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/trm")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina"])
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit training samples (for testing)")

    # TRM specific
    parser.add_argument("--n_latent", type=int, default=6, help="Level 3: n")
    parser.add_argument("--T_recursion", type=int, default=3, help="Level 2: T")
    parser.add_argument("--N_supervision", type=int, default=16, help="Level 1: N_sup")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")

    # Performance optimizations
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (bfloat16)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable AMP, use float32")

    args = parser.parse_args()

    print(f"Loading QwenTRM with backbone: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TRM Configuration
    trm_config = TRMConfig(
        n_latent=args.n_latent,
        T_recursion=args.T_recursion,
        N_supervision=args.N_supervision
    )

    # Load QwenTRM
    model = QwenTRM.from_pretrained_backbone(
        backbone_name=args.model,
        config=trm_config,
        device="cuda",
        init_lm_head=True
    )

    # Convert TRM components to bfloat16 for AMP compatibility
    # This avoids dtype mismatch warnings and enables fused kernels
    if args.amp:
        print("[QwenTRM] Converting TRM components to bfloat16...")
        model.interface = model.interface.to(torch.bfloat16)
        model.engine = model.engine.to(torch.bfloat16)
        model.heads = model.heads.to(torch.bfloat16)

    # Optional: torch.compile for speedup
    if args.compile:
        print("Compiling TRM engine with torch.compile...")
        # Use default mode instead of reduce-overhead to avoid CUDAGraph issues
        # with TRM's recursive structure
        model.engine = torch.compile(model.engine, mode="default")

    trainable, total = count_trainable_params(model)
    print(f"\nTRM Configuration:")
    print(f"  n_latent (Level 3): {args.n_latent}")
    print(f"  T_recursion (Level 2): {args.T_recursion}")
    print(f"  N_supervision (Level 1): {args.N_supervision}")
    print(f"  Effective depth: {2 * (args.n_latent + 1) * args.T_recursion * args.N_supervision}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  torch.compile: {args.compile}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
    else:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    # Create datasets with pre-tokenization
    train_dataset = MathDataset(dataset, tokenizer, args.max_length)

    # Optimized DataLoader (fixed-length padding, no custom collate needed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # Single process to avoid multiprocessing issues
        pin_memory=True,
        drop_last=True          # Avoid uneven final batch
    )

    # Training config with AMP
    training_config = TrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        log_steps=10,
        save_steps=500,
        eval_steps=100,
        use_amp=args.amp,
        amp_dtype="bfloat16"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config
    )

    print("\nStarting TRM training...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mixed Precision (AMP): {args.amp} {'(bfloat16)' if args.amp else '(float32)'}")

    trainer.train()

    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
