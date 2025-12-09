"""
GSM8K Evaluation Script
Phase 1: Qwen Baseline + TRM Identity Test
"""

import argparse
import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text: str) -> int:
    """Extract numeric answer from model output.

    Handles multiple formats:
    - #### 12345
    - The answer is 12345
    - \\boxed{12345}
    - Final answer: 12345
    """
    # Try #### pattern first (GSM8K format)
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try \\boxed{} pattern
    match = re.search(r"\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}", text)
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


def format_prompt(question: str) -> str:
    """Format question into chat prompt for Qwen."""
    return f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""


def evaluate_qwen_baseline(
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
    num_samples: int = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    device: str = "cuda"
):
    """Evaluate Qwen model on GSM8K test set."""

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

        prompt = format_prompt(question)

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

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(prompt):]

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

    args = parser.parse_args()

    results = evaluate_qwen_baseline(
        model_name=args.model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device
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
