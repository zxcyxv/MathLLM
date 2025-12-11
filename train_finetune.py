"""
Last N Layers Finetuning Script for Qwen-2.5-Math
Baseline for TRM comparison

Unfreeze the last N transformer layers + lm_head for fair comparison with TRM.
Uses identical data preprocessing as train_trm.py (ChatML format, \boxed{}).
"""

import argparse
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def convert_to_boxed(answer: str) -> str:
    """Convert GSM8K '#### 72' format to '\\boxed{72}' format."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*)', answer)
    if match:
        num = match.group(1)
        old = match.group(0)
        new = '\\boxed{' + num + '}'
        answer = answer.replace(old, new)
    return answer


class FinetuneDataset(torch.utils.data.Dataset):
    """
    Dataset for finetuning with ChatML format.
    Identical preprocessing to train_trm.py for fair comparison.
    """

    def __init__(self, dataset, tokenizer, max_length=1024,
                 question_col="question", answer_col="answer", convert_format=True):
        self.max_length = max_length
        self.tokenizer = tokenizer

        print(f"Pre-tokenizing dataset with ChatML format...")
        print(f"  Question column: {question_col}, Answer column: {answer_col}")
        print(f"  Convert format (#### -> \\boxed): {convert_format}")
        self.tokenized_data = []

        for example in tqdm(dataset, desc="Tokenizing"):
            question = example[question_col]
            answer = example[answer_col]

            # Convert GSM8K format if needed
            if convert_format:
                answer = convert_to_boxed(answer)

            # Create ChatML messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]

            # Full text (includes assistant response)
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Prompt only (for masking)
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            full_tokens = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length
            )["input_ids"]

            prompt_tokens = tokenizer(
                prompt_text,
                add_special_tokens=False
            )["input_ids"]

            # Create labels (mask prompt, keep response)
            labels = full_tokens.copy()
            prompt_len = len(prompt_tokens)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            self.tokenized_data.append({
                "input_ids": full_tokens,
                "labels": labels,
                "attention_mask": [1] * len(full_tokens)
            })

        print(f"Tokenized {len(self.tokenized_data)} examples")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long)
        }


def reinit_layer(layer):
    """Reinitialize a transformer layer's weights."""
    for name, param in layer.named_parameters():
        if 'weight' in name:
            if 'norm' in name or 'ln' in name:
                # LayerNorm/RMSNorm: init to 1
                torch.nn.init.ones_(param)
            elif len(param.shape) >= 2:
                # Linear layers: use xavier/glorot
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)


def freeze_except_last_n_layers(model, n_layers: int, include_lm_head: bool = True, reinit: bool = False):
    """
    Freeze all parameters except last N transformer layers.

    Qwen structure:
    - model.embed_tokens
    - model.layers[0..27]  (28 layers total)
    - model.norm
    - lm_head

    Args:
        model: The model
        n_layers: Number of last layers to unfreeze
        include_lm_head: Also unfreeze lm_head
        reinit: Reinitialize unfrozen layers (for fair comparison with TRM)
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
        # Reinitialize if requested
        if reinit:
            reinit_layer(layers[i])
            print(f"  [REINIT] Layer {i} weights reinitialized")

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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Number of last layers to unfreeze")
    parser.add_argument("--include_lm_head", action="store_true", default=False,
                        help="Also unfreeze lm_head (adds ~545M params)")
    parser.add_argument("--lm_head_only", action="store_true", default=False,
                        help="Only unfreeze lm_head (ignores n_layers)")
    parser.add_argument("--reinit_layers", action="store_true", default=False,
                        help="Reinitialize unfrozen layers (for fair comparison with TRM)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)  # Lower LR for finetuning
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina", "math"],
                        help="Dataset: gsm8k (7.5K), numina (860K), math (7.5K)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit training samples (for testing)")

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
            args.include_lm_head,
            reinit=args.reinit_layers
        )

    trainable, total = count_params(model)
    print(f"\nFinetuning Configuration:")
    print(f"  Total layers: {total_layers}")
    print(f"  Unfrozen layers: {start_layer} to {total_layers-1} ({args.n_layers} layers)")
    print(f"  Include lm_head: {args.include_lm_head}")
    print(f"  Reinit layers: {args.reinit_layers}")
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Total params: {total:,}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
        question_col, answer_col = "question", "answer"
    elif args.dataset == "numina":
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        question_col, answer_col = "problem", "solution"
    elif args.dataset == "math":
        dataset = load_dataset("hendrycks/competition_math", split="train")
        question_col, answer_col = "problem", "solution"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    # Preprocess (same as train_trm.py)
    convert_format = (args.dataset == "gsm8k")
    train_dataset = FinetuneDataset(
        dataset, tokenizer, args.max_length,
        question_col=question_col,
        answer_col=answer_col,
        convert_format=convert_format
    )

    # Data collator for padding
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), tokenizer.pad_token_id)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask)
        }

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
        data_collator=collate_fn,
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
