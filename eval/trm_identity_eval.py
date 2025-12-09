"""
TRM Identity Test Evaluation
Phase 1.3: Verify TRM initialization doesn't degrade Qwen baseline
"""

import argparse
import re
import json
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/MathLLM")
from src.model import QwenTRM
from src.config import TRMConfig


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


def evaluate_trm_identity(
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
    num_samples: int = None,
    max_new_tokens: int = 512,
    num_supervision_steps: int = 1,  # Minimal steps for identity test
    device: str = "cuda"
):
    """Evaluate QwenTRM (untrained) on GSM8K."""

    print(f"Loading QwenTRM with backbone: {model_name}")
    print(f"Supervision steps: {num_supervision_steps}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load QwenTRM
    config = TRMConfig()
    model = QwenTRM.from_pretrained_backbone(
        backbone_name=model_name,
        config=config,
        device=device,
        init_lm_head=True  # Use SVD-compressed Qwen lm_head
    )
    model.eval()

    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")

    correct = 0
    total = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating TRM"):
        question = item["question"]
        answer_text = item["answer"]
        ground_truth = extract_ground_truth(answer_text)

        if ground_truth is None:
            continue

        prompt = format_prompt(question)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Generate using QwenTRM
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                num_supervision_steps=num_supervision_steps
            )

        response = tokenizer.decode(generated[0], skip_special_tokens=True)
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
            "response": response_only[:500]
        })

        if total % 50 == 0:
            print(f"Progress: {total}/{len(dataset)}, Accuracy: {correct/total*100:.2f}%")

    accuracy = correct / total * 100

    print("\n" + "=" * 50)
    print(f"Model: QwenTRM (backbone: {model_name})")
    print(f"Supervision Steps: {num_supervision_steps}")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

    return {
        "model": f"QwenTRM({model_name})",
        "supervision_steps": num_supervision_steps,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="TRM Identity Test on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--supervision_steps", type=int, default=1,
                        help="Number of supervision steps (1 = minimal)")
    parser.add_argument("--output", type=str, default="trm_identity_results.json")

    args = parser.parse_args()

    results = evaluate_trm_identity(
        model_name=args.model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        num_supervision_steps=args.supervision_steps
    )

    with open(args.output, "w") as f:
        json.dump({
            "model": results["model"],
            "supervision_steps": results["supervision_steps"],
            "total": results["total"],
            "correct": results["correct"],
            "accuracy": results["accuracy"]
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
