"""
GSM8K Evaluation Script - Batch Version (Faster)
Uses batched inference for speed improvement
"""

import argparse
import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text: str) -> int:
    """Extract numeric answer from model output."""
    # Try #### pattern first
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try \\boxed{}
    match = re.search(r"\\boxed\{(-?\d+(?:,\d+)*(?:\.\d+)?)\}", text)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Try "answer is"
    match = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return int(float(match.group(1).replace(",", "")))

    # Last number
    numbers = re.findall(r"(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if numbers:
        return int(float(numbers[-1].replace(",", "")))

    return None


def extract_ground_truth(answer_text: str) -> int:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
    if match:
        return int(float(match.group(1).replace(",", "")))
    return None


def format_prompt(question: str) -> str:
    return f"""Solve this math problem step by step. Show your work and put your final answer after ####.

Question: {question}

Solution:"""


def evaluate_batch(
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
    num_samples: int = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    device: str = "cuda"
):
    """Evaluate with batched inference."""

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
    tokenizer.padding_side = "left"  # Important for batch generation

    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples with batch_size={batch_size}...")

    # Prepare all data
    questions = [item["question"] for item in dataset]
    answers = [item["answer"] for item in dataset]
    ground_truths = [extract_ground_truth(a) for a in answers]
    prompts = [format_prompt(q) for q in questions]

    correct = 0
    total = 0
    results = []

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i:i+batch_size]
        batch_gts = ground_truths[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode and evaluate each in batch
        for j, (output, prompt, gt, question) in enumerate(zip(outputs, batch_prompts, batch_gts, batch_questions)):
            if gt is None:
                continue

            response = tokenizer.decode(output, skip_special_tokens=True)
            response_only = response[len(prompt):]

            predicted = extract_answer(response_only)
            is_correct = (predicted == gt)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": question,
                "ground_truth": gt,
                "predicted": predicted,
                "correct": is_correct
            })

        # Progress report
        if (i // batch_size + 1) % 10 == 0:
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
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K (Batch)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default="gsm8k_results.json")

    args = parser.parse_args()

    results = evaluate_batch(
        model_name=args.model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )

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
