"""
LoRA Training Script for Qwen-2.5-Math-7B
Phase 2: Control group for TRM comparison

Goal: Match ~160M trainable params with TRM for fair comparison
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


def format_example(example: dict) -> str:
    """Format GSM8K example for training."""
    question = example["question"]
    answer = example["answer"]
    return f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution: {answer}"""


def preprocess_dataset(dataset, tokenizer, max_length=1024):
    """Tokenize dataset for causal LM training."""

    def tokenize_fn(examples):
        texts = [format_example({"question": q, "answer": a})
                 for q, a in zip(examples["question"], examples["answer"])]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )


def count_trainable_params(model):
    """Count trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with LoRA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/lora")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina"])
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit training samples (for testing)")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # LoRA config targeting attention layers
    # Target: q_proj, k_proj, v_proj, o_proj (like paper comparisons)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    trainable, total = count_trainable_params(model)
    print(f"\nLoRA Configuration:")
    print(f"  Rank: {args.rank}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Total params: {total:,}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
    else:
        # NuminaMath for harder problems
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    # Preprocess
    train_dataset = preprocess_dataset(dataset, tokenizer, args.max_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=False,  # Disabled for LoRA compatibility
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
