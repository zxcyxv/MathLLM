"""
TRM Training Script for Qwen-2.5-Math-7B + TRM
Phase 2: Experimental group for LoRA comparison

Implements the 3-level recursive training from TRM paper.

CRITICAL: Uses Qwen's official ChatML format via apply_chat_template().
This ensures the model sees the exact same format it was pretrained on.

Key tokens (Qwen2.5-Math):
- <|im_start|> (ID: 151644): Message start
- <|im_end|> (ID: 151645): Message end / EOS token
- <|endoftext|> (ID: 151643): PAD token
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


# System prompt for math reasoning (Qwen-Math default style)
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def convert_to_boxed(answer: str) -> str:
    """Convert GSM8K '#### 72' format to '\\boxed{72}' format.

    This aligns training data with Qwen-Math's expected output format.
    """
    import re
    # Replace "#### 123" with "\\boxed{123}"
    match = re.search(r'####\s*(-?\d+(?:,\d+)*)', answer)
    if match:
        num = match.group(1)
        # Use string replace instead of re.sub to avoid backslash issues
        old = match.group(0)  # "#### 72"
        new = '\\\\boxed{' + num + '}'  # "\\boxed{72}"
        answer = answer.replace(old, new)
    return answer


def create_messages(question: str, answer: str = None, convert_format: bool = True) -> list:
    """Create ChatML message format for Qwen.

    Args:
        question: The math problem
        answer: The solution (None for inference)
        convert_format: If True, convert GSM8K '#### N' to '\\boxed{N}' format.
                       Set False for datasets already using \\boxed{} (numina, math)

    Returns:
        List of message dicts for apply_chat_template
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        # Only convert GSM8K format (#### N -> \boxed{N})
        if convert_format:
            answer = convert_to_boxed(answer)
        messages.append({"role": "assistant", "content": answer})
    return messages


class MathDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for math problems using Qwen's ChatML format.

    CRITICAL fixes for proper training:
    1. Uses tokenizer.apply_chat_template() for exact ChatML format
    2. <|im_end|> token at response end (teaches model when to stop)
    3. System + User portions masked in labels (only learn assistant response)
    4. Identical format between training and inference
    """

    def __init__(self, dataset, tokenizer, max_length=1024,
                 question_col="question", answer_col="answer", convert_format=True):
        """
        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer
            max_length: Max sequence length
            question_col: Column name for questions (gsm8k: 'question', numina/math: 'problem')
            answer_col: Column name for answers (gsm8k: 'answer', numina/math: 'solution')
            convert_format: If True, convert '#### N' to '\\boxed{N}' (only for gsm8k)
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Pre-tokenize all examples
        print(f"Pre-tokenizing dataset with ChatML format...")
        print(f"  Question column: {question_col}, Answer column: {answer_col}")
        print(f"  Convert format (#### -> \\boxed): {convert_format}")
        self.tokenized_data = []

        for example in tqdm(dataset, desc="Tokenizing"):
            question = example[question_col]
            answer = example[answer_col]

            # Create full conversation (system + user + assistant)
            messages = create_messages(question, answer, convert_format=convert_format)

            # Get the full formatted text (for training, includes assistant response)
            # add_generation_prompt=False because we include the assistant message
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Get prompt-only text (system + user + "assistant\n")
            # This is what we mask in labels
            prompt_messages = create_messages(question, answer=None, convert_format=False)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True  # Adds "<|im_start|>assistant\n"
            )

            # Tokenize prompt to get its length (for masking)
            prompt_tokens = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length
            )
            prompt_len = len(prompt_tokens["input_ids"])

            # Tokenize full text
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)

            # Labels: mask prompt (system + user + assistant header) AND padding
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # Don't learn to predict prompt
            labels[attention_mask == 0] = -100  # Don't learn padding

            self.tokenized_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        print(f"Pre-tokenized {len(self.tokenized_data)} examples with ChatML format")

        # Debug: show one example
        if len(self.tokenized_data) > 0:
            example = self.tokenized_data[0]
            decoded = tokenizer.decode(example["input_ids"][:100], skip_special_tokens=False)
            print(f"[Debug] First example (first 100 tokens): {repr(decoded[:200])}")

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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/trm")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina", "math"],
                        help="Dataset: gsm8k (7.5K), numina (860K), math (7.5K)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit training samples (for testing)")

    # TRM specific
    parser.add_argument("--n_latent", type=int, default=6, help="Level 3: n")
    parser.add_argument("--T_recursion", type=int, default=3, help="Level 2: T")
    parser.add_argument("--N_supervision", type=int, default=16, help="Level 1: N_sup")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")

    # DIS (Deep Improvement Supervision) specific
    parser.add_argument("--use_dis", action="store_true",
                        help="Enable DIS mode (progressive target supervision)")
    parser.add_argument("--dis_noise_schedule", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Noise schedule for DIS target corruption")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")

    # Performance optimizations
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (bfloat16)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable AMP, use float32")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., ./checkpoints/trm/checkpoint-234)")

    # Freeze options
    parser.add_argument("--freeze_lm_head", action="store_true",
                        help="Freeze lm_head (only train TRM block)")

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
        N_supervision=args.N_supervision,
        use_dis=args.use_dis,
        dis_noise_schedule=args.dis_noise_schedule
    )

    # Load QwenTRM
    model = QwenTRM.from_pretrained_backbone(
        backbone_name=args.model,
        config=trm_config,
        device="cuda",
        init_lm_head=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        ckpt_file = os.path.join(args.resume, "trm_model.pt")
        if os.path.exists(ckpt_file):
            print(f"[Resume] Loading checkpoint from: {ckpt_file}")
            state = torch.load(ckpt_file, map_location="cuda", weights_only=False)
            model.interface.load_state_dict(state['interface'])
            model.engine.load_state_dict(state['engine'])
            model.heads.load_state_dict(state['heads'])
            start_epoch = state.get('epoch', 0)
            print(f"[Resume] Loaded checkpoint (epoch={start_epoch}, step={state.get('global_step', '?')})")
        else:
            print(f"[Resume] Warning: Checkpoint not found at {ckpt_file}, starting fresh")

    # Freeze lm_head if requested
    if args.freeze_lm_head:
        print("[QwenTRM] Freezing lm_head (will not be trained)")
        for param in model.heads.lm_head.parameters():
            param.requires_grad = False

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
    if trm_config.use_dis:
        print(f"  Mode: DIS (Deep Improvement Supervision)")
        print(f"  n_latent (Level 3): {trm_config.active_n_latent} (DIS: {trm_config.dis_n_latent}, Standard: {trm_config.n_latent})")
        print(f"  T_recursion (Level 2): {trm_config.active_T_recursion} (DIS: {trm_config.dis_T_recursion}, Standard: {trm_config.T_recursion})")
        print(f"  N_supervision (Level 1): {trm_config.active_N_supervision} (DIS: {trm_config.dis_N_supervision}, Standard: {trm_config.N_supervision})")
        print(f"  Noise schedule: {trm_config.dis_noise_schedule}")
        print(f"  Effective depth: {2 * (trm_config.active_n_latent + 1) * trm_config.active_T_recursion * trm_config.active_N_supervision} (vs {2 * (trm_config.n_latent + 1) * trm_config.T_recursion * trm_config.N_supervision} in standard TRM)")
    else:
        print(f"  Mode: Standard TRM")
        print(f"  n_latent (Level 3): {trm_config.n_latent}")
        print(f"  T_recursion (Level 2): {trm_config.T_recursion}")
        print(f"  N_supervision (Level 1): {trm_config.N_supervision}")
        print(f"  Effective depth: {2 * (trm_config.n_latent + 1) * trm_config.T_recursion * trm_config.N_supervision}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  torch.compile: {args.compile}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
        # GSM8K uses 'question' and 'answer' columns
        question_col, answer_col = "question", "answer"
    elif args.dataset == "numina":
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        # NuminaMath uses 'problem' and 'solution' columns (already \boxed{} format)
        question_col, answer_col = "problem", "solution"
    elif args.dataset == "math":
        dataset = load_dataset("hendrycks/competition_math", split="train")
        # MATH uses 'problem' and 'solution' columns (already \boxed{} format)
        question_col, answer_col = "problem", "solution"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Training samples: {len(dataset)}")

    # Create datasets with pre-tokenization
    # Only GSM8K needs format conversion (#### -> \boxed)
    convert_format = (args.dataset == "gsm8k")
    train_dataset = MathDataset(
        dataset, tokenizer, args.max_length,
        question_col=question_col,
        answer_col=answer_col,
        convert_format=convert_format
    )

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
