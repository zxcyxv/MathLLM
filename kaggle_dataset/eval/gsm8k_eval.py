"""
GSM8K Evaluation Script
Phase 1: Qwen Baseline + TRM Identity Test

CRITICAL: Uses Qwen's official ChatML format via apply_chat_template().
"""

import argparse
import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# System prompt - same as training for consistency
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_answer(text: str) -> int:
    """Extract numeric answer from model output.

    Handles multiple formats:
    - \\boxed{12345} (Qwen-Math style, check first)
    - #### 12345 (GSM8K format)
    - The answer is 12345
    """
    # Try \\boxed{} pattern first (Qwen-Math default)
    match = re.search(r"\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try #### pattern (GSM8K format)
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try "answer is" pattern
    match = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try to find the last number in the text
    numbers = re.findall(r"(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if numbers:
        return int(float(numbers[-1].replace(",", "")))

    return None


def extract_ground_truth(answer_text: str) -> int:
    """Extract ground truth from GSM8K answer format."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
    if match:
        return int(float(match.group(1).replace(",", "")))
    return None


def create_messages(question: str) -> list:
    """Create ChatML message format for Qwen."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def print_sample_output(idx: int, question: str, response: str, predicted: int,
                        ground_truth: int, is_correct: bool):
    """Print detailed output for a single sample."""
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    print("\n" + "─" * 60)
    print(f"[Sample {idx}] {status}")
    print("─" * 60)
    print(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}")
    print(f"\nModel Output:")
    print("-" * 40)
    print(response[:1000] if response else "(empty)")
    if len(response) > 1000:
        print(f"... (truncated, total {len(response)} chars)")
    print("-" * 40)
    print(f"Predicted: {predicted} | Ground Truth: {ground_truth}")
    print("─" * 60)


def evaluate_qwen_baseline(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    num_samples: int = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: str = "cuda",
    verbose: bool = False,
    show_wrong_only: bool = False
):
    """Evaluate Qwen model on GSM8K test set using proper ChatML format."""

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # DON'T override pad_token - use Qwen's defaults
    print(f"EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")

    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")

    correct = 0
    total = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        answer_text = item["answer"]
        ground_truth = extract_ground_truth(answer_text)

        if ground_truth is None:
            continue

        # Use ChatML format via apply_chat_template
        messages = create_messages(question)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Adds "<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Only decode the NEW tokens (after input_ids)
        input_len = inputs['input_ids'].size(1)
        response_only = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

        predicted = extract_answer(response_only)

        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "response": response_only[:500]  # Truncate for storage
        })

        # Verbose output
        if verbose and (not show_wrong_only or not is_correct):
            print_sample_output(
                idx=total,
                question=question,
                response=response_only,
                predicted=predicted,
                ground_truth=ground_truth,
                is_correct=is_correct
            )

        if total % 50 == 0:
            print(f"Progress: {total}/{len(dataset)}, Accuracy: {correct/total*100:.2f}%")

    accuracy = correct / total * 100

    print("\n" + "=" * 50)
    print(f"Model: {model_name}")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--output", type=str, default="gsm8k_results.json",
                        help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed model outputs for each sample")
    parser.add_argument("--wrong-only", action="store_true",
                        help="Only show outputs for wrong predictions (requires --verbose)")

    args = parser.parse_args()

    results = evaluate_qwen_baseline(
        model_name=args.model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        verbose=args.verbose,
        show_wrong_only=args.wrong_only
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "model": results["model"],
            "total": results["total"],
            "correct": results["correct"],
            "accuracy": results["accuracy"]
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
