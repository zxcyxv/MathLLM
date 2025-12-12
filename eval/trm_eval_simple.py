"""
TRM Evaluation Script (Simple Version)
Based on the working manual test code.
"""

import argparse
import re
import json
import sys
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/MathLLM")
from src.model import QwenTRM
from src.config import TRMConfig


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_answer(text: str):
    """Extract numeric answer from model output."""
    # Try \\boxed{} first
    match = re.search(r"\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try #### pattern
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try "answer is"
    match = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Last number as fallback
    numbers = re.findall(r"(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if numbers:
        try:
            val = float(numbers[-1].replace(",", ""))
            if abs(val) < 1e15:
                return int(val)
        except:
            pass

    return None


def extract_ground_truth(answer_text: str):
    """Extract ground truth from GSM8K answer format."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
    if match:
        return int(float(match.group(1).replace(",", "")))
    return None


def extract_ground_truth_boxed(answer_text: str):
    """Extract ground truth from \\boxed{} format (NuminaMath, MATH)."""
    # Find the last \boxed{} in the text
    matches = re.findall(r"\\boxed\{([^}]+)\}", answer_text)
    if matches:
        # Take the last match (final answer)
        val = matches[-1].strip()
        # Try to parse as number
        try:
            # Remove commas and parse
            val_clean = val.replace(",", "").replace(" ", "")
            return int(float(val_clean))
        except:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM on Math Datasets")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--supervision_steps", type=int, default=16)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--wrong_only", action="store_true")
    parser.add_argument("--output", type=str, default="trm_eval_results.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina", "math"],
                        help="Dataset: gsm8k, numina (NuminaMath-CoT), math (MATH)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N samples (useful for testing on unseen data)")

    args = parser.parse_args()
    device = "cuda"

    # Load tokenizer (no special settings needed)
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load model
    config = TRMConfig()
    model = QwenTRM.from_pretrained_backbone(args.model, config, device, init_lm_head=True)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt_file = os.path.join(args.checkpoint, "trm_model.pt")
        print(f"Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.interface.load_state_dict(ckpt['interface'])
        model.engine.load_state_dict(ckpt['engine'])
        model.heads.load_state_dict(ckpt['heads'])
        print(f"Loaded checkpoint (step={ckpt.get('global_step', '?')})")

    # Convert to bfloat16
    model.interface = model.interface.to(torch.bfloat16)
    model.engine = model.engine.to(torch.bfloat16)
    model.heads = model.heads.to(torch.bfloat16)
    model.eval()

    # Load dataset
    print(f"Loading {args.dataset} {args.split} set...")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=args.split)
        question_col, answer_col = "question", "answer"
        use_boxed_format = False
    elif args.dataset == "numina":
        # NuminaMath only has train split
        split = "train" if args.split == "test" else args.split
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split=split)
        question_col, answer_col = "problem", "solution"
        use_boxed_format = True
    elif args.dataset == "math":
        dataset = load_dataset("hendrycks/competition_math", split=args.split)
        question_col, answer_col = "problem", "solution"
        use_boxed_format = True
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Apply offset and num_samples
    if args.offset > 0:
        end_idx = len(dataset)
        if args.num_samples:
            end_idx = min(args.offset + args.num_samples, len(dataset))
        dataset = dataset.select(range(args.offset, end_idx))
    elif args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Evaluating {len(dataset)} samples...")

    correct = 0
    total = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        question = item[question_col]
        answer_text = item[answer_col]

        # Extract ground truth based on format
        if use_boxed_format:
            ground_truth = extract_ground_truth_boxed(answer_text)
        else:
            ground_truth = extract_ground_truth(answer_text)

        if ground_truth is None:
            continue

        # Create prompt (same as working manual test)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize (simple, no padding, no attention_mask passed)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Generate (same as working manual test)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                num_supervision_steps=args.supervision_steps,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(gen[0, input_ids.size(1):], skip_special_tokens=True)
        predicted = extract_answer(response)

        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:500]
        })

        # Verbose output
        if args.verbose and (not args.wrong_only or not is_correct):
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"\n{'═'*80}")
            print(f"[Sample {total}] {status}")
            print(f"Predicted: {predicted} | Ground Truth: {ground_truth}")
            print(f"{'─'*80}")
            print(f"QUESTION:")
            print(question)
            print(f"{'─'*80}")
            print(f"MODEL OUTPUT:")
            print(response)
            print(f"{'═'*80}")

        # Progress
        if total % 50 == 0:
            print(f"Progress: {total}/{len(dataset)}, Acc: {correct/total*100:.1f}%")

    # Final results
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 50)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "checkpoint": args.checkpoint,
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
