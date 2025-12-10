"""
Last N Layers Finetuning Script for Qwen-2.5-Math-7B
Phase 2: Control group for TRM comparison

Instead of LoRA, unfreeze the last N transformer layers for fair comparison.
This matches the "additional computation" paradigm more closely.
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

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )


def freeze_except_last_n_layers(model, n_layers: int, include_lm_head: bool = True):
    """
    Freeze all parameters except last N transformer layers.

    Qwen structure:
    - model.embed_tokens
    - model.layers[0..27]  (28 layers total)
    - model.norm
    - lm_head
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Get number of layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        total_layers = len(layers)
    elif hasattr(model, 'layers'):
        layers = model.layers
        total_layers = len(layers)
    else:
        raise ValueError("Cannot find transformer layers in model")

    # Unfreeze last N layers
    start_layer = total_layers - n_layers
    for i in range(start_layer, total_layers):
        for param in layers[i].parameters():
            param.requires_grad = True

    # Unfreeze final norm
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        for param in model.model.norm.parameters():
            param.requires_grad = True
    elif hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True

    # Optionally unfreeze lm_head
    if include_lm_head and hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True

    return start_layer, total_layers


def count_params(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with Last N Layers Unfrozen")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Number of last layers to unfreeze")
    parser.add_argument("--include_lm_head", action="store_true", default=False,
                        help="Also unfreeze lm_head (adds ~545M params)")
    parser.add_argument("--lm_head_only", action="store_true", default=False,
                        help="Only unfreeze lm_head (ignores n_layers)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)  # Lower LR for finetuning
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina"])
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Freeze except specified layers
    if args.lm_head_only:
        # Only unfreeze lm_head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
        start_layer, total_layers = -1, 28
    else:
        start_layer, total_layers = freeze_except_last_n_layers(
            model,
            args.n_layers,
            args.include_lm_head
        )

    trainable, total = count_params(model)
    print(f"\nFinetuning Configuration:")
    print(f"  Total layers: {total_layers}")
    print(f"  Unfrozen layers: {start_layer} to {total_layers-1} ({args.n_layers} layers)")
    print(f"  Include lm_head: {args.include_lm_head}")
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Total params: {total:,}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
    else:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    # Preprocess
    train_dataset = preprocess_dataset(dataset, tokenizer, args.max_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
        gradient_checkpointing=False,
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

    # Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
