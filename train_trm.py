"""
TRM Training Script for Qwen-2.5-Math-7B + TRM
Phase 2: Experimental group for LoRA comparison

Implements the 3-level recursive training from TRM paper.
"""

import argparse
import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
    """Dataset wrapper for math problems."""

    def __init__(self, dataset, tokenizer, max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = format_example(example)

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Mask padding tokens in labels with -100 (ignored by CrossEntropyLoss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


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

    # Optional: torch.compile for speedup
    if args.compile:
        print("Compiling TRM engine with torch.compile...")
        model.engine = torch.compile(model.engine, mode="reduce-overhead")

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

    # Create datasets
    train_dataset = MathDataset(dataset, tokenizer, args.max_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )

    # Training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        log_steps=10,
        save_steps=500,
        eval_steps=100
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

    trainer.train()

    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
