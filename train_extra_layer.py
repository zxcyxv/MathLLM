"""
Extra Layer Training Script for Qwen-2.5-Math
Baseline for TRM comparison - adds a new Qwen2DecoderLayer after the backbone.

This is a fair comparison with TRM:
- TRM: Qwen 28층 (frozen) → TRM Block (new, random init) → lm_head
- This: Qwen 28층 (frozen) → Qwen2DecoderLayer (new, random init) → lm_head
"""

import argparse
import re
import copy
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


class QwenWithExtraLayer(nn.Module):
    """
    Qwen model with an additional decoder layer inserted before lm_head.

    Architecture:
        Qwen Backbone (28 layers, frozen)
        → Extra Qwen2DecoderLayer (new, random init, trainable)
        → Final Norm (frozen or trainable)
        → lm_head (trainable)
    """

    def __init__(self, base_model, train_lm_head=True):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # Freeze backbone
        for param in base_model.parameters():
            param.requires_grad = False

        # Create a new decoder layer (copy structure, reinitialize weights)
        original_layer = base_model.model.layers[-1]
        self.extra_layer = self._create_new_layer(original_layer)

        # Extra layer's norm (copy from final norm, reinitialize)
        self.extra_norm = copy.deepcopy(base_model.model.norm)
        self._reinit_norm(self.extra_norm)
        for param in self.extra_norm.parameters():
            param.requires_grad = True

        # Unfreeze lm_head if requested
        if train_lm_head:
            for param in base_model.lm_head.parameters():
                param.requires_grad = True

        # Count params
        self._print_param_info()

    def _create_new_layer(self, original_layer):
        """Create a new decoder layer with random initialization."""
        # Deep copy the structure
        new_layer = copy.deepcopy(original_layer)

        # Reinitialize all weights
        for name, param in new_layer.named_parameters():
            param.requires_grad = True
            if 'weight' in name:
                if 'norm' in name or 'layernorm' in name:
                    nn.init.ones_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        return new_layer

    def _reinit_norm(self, norm):
        """Reinitialize RMSNorm."""
        if hasattr(norm, 'weight'):
            nn.init.ones_(norm.weight)

    def _print_param_info(self):
        """Print parameter information."""
        extra_params = sum(p.numel() for p in self.extra_layer.parameters())
        norm_params = sum(p.numel() for p in self.extra_norm.parameters())
        lm_head_params = sum(p.numel() for p in self.base_model.lm_head.parameters() if p.requires_grad)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(f"\n=== QwenWithExtraLayer ===")
        print(f"  Extra layer params: {extra_params:,}")
        print(f"  Extra norm params: {norm_params:,}")
        print(f"  lm_head params: {lm_head_params:,}")
        print(f"  Total trainable: {trainable:,}")
        print(f"  Total params: {total:,}")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get hidden states from backbone (without lm_head)
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state

        # Apply extra layer
        # Qwen2DecoderLayer expects: (hidden_states, attention_mask, position_ids, ...)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position_ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Forward through extra layer
        layer_outputs = self.extra_layer(
            hidden_states,
            attention_mask=None,  # Causal mask is handled internally
            position_ids=position_ids,
        )
        hidden_states = layer_outputs[0]

        # Apply final norm
        hidden_states = self.extra_norm(hidden_states)

        # Apply lm_head
        logits = self.base_model.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()

    def generate(self, input_ids, **kwargs):
        """Generate using the model."""
        # For generation, we need a custom approach
        # Simple greedy decoding
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        eos_token_id = kwargs.get('eos_token_id', None)

        device = input_ids.device
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


class ExtraLayerDataset(torch.utils.data.Dataset):
    """Dataset for extra layer training with ChatML format."""

    def __init__(self, dataset, tokenizer, max_length=1024,
                 question_col="question", answer_col="answer", convert_format=True):
        self.max_length = max_length
        self.tokenizer = tokenizer

        print(f"Pre-tokenizing dataset with ChatML format...")
        self.tokenized_data = []

        for example in tqdm(dataset, desc="Tokenizing"):
            question = example[question_col]
            answer = example[answer_col]

            if convert_format:
                answer = convert_to_boxed(answer)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            full_tokens = tokenizer(
                full_text, add_special_tokens=False,
                truncation=True, max_length=max_length
            )["input_ids"]

            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            labels = full_tokens.copy()
            for i in range(min(len(prompt_tokens), len(labels))):
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


def collate_fn(batch, pad_token_id):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with Extra Layer (TRM baseline)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/extra_layer")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina", "math"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--freeze_lm_head", action="store_true",
                        help="Freeze lm_head (only train extra layer)")

    args = parser.parse_args()
    device = "cuda"

    print(f"Loading model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Create model with extra layer
    model = QwenWithExtraLayer(base_model, train_lm_head=not args.freeze_lm_head)

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

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    convert_format = (args.dataset == "gsm8k")
    train_dataset = ExtraLayerDataset(
        dataset, tokenizer, args.max_length,
        question_col=question_col,
        answer_col=answer_col,
        convert_format=convert_format
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    'loss': f'{epoch_loss/(step+1):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

        print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss/len(train_loader):.4f}")

    # Save
    print(f"\nSaving model to {args.output_dir}")
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Save extra layer and lm_head
    torch.save({
        'extra_layer': model.extra_layer.state_dict(),
        'extra_norm': model.extra_norm.state_dict(),
        'lm_head': model.base_model.lm_head.state_dict(),
        'config': {
            'base_model': args.model,
            'epochs': args.epochs,
            'lr': args.lr,
        }
    }, os.path.join(args.output_dir, 'extra_layer_model.pt'))

    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
