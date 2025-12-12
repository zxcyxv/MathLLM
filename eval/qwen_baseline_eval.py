"""
Qwen Baseline Evaluation Script
기본 Qwen 모델의 GSM8K 성능을 측정합니다.
TRM 모델과 공정한 비교를 위해 동일한 평가 로직 사용.

사용법:
    python eval/qwen_baseline_eval.py --model Qwen/Qwen2.5-Math-7B-Instruct
    python eval/qwen_baseline_eval.py --model Qwen/Qwen2.5-Math-1.5B-Instruct -v
"""

import argparse
import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_answer(text: str):
    """Extract numeric answer from model output.
    TRM 평가와 동일한 로직 사용.
    """
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
    """Extract ground truth from GSM8K answer format.
    TRM 평가와 동일한 로직 사용.
    """
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
    if match:
        return int(float(match.group(1).replace(",", "")))
    return None


def extract_ground_truth_boxed(answer_text: str):
    """Extract ground truth from \\boxed{} format (NuminaMath, MATH)."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", answer_text)
    if matches:
        val = matches[-1].strip()
        try:
            val_clean = val.replace(",", "").replace(" ", "")
            return int(float(val_clean))
        except:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Qwen on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed output for each sample")
    parser.add_argument("--wrong_only", action="store_true",
                        help="Only show wrong answers in verbose mode")
    parser.add_argument("--output", type=str, default="qwen_baseline_results.json",
                        help="Output JSON file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "numina", "math"],
                        help="Dataset: gsm8k, numina (NuminaMath-CoT), math (MATH)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N samples (useful for testing on unseen data)")

    args = parser.parse_args()
    device = "cuda"

    # =========================================================================
    # 1. Load Tokenizer
    # =========================================================================
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # =========================================================================
    # 2. Load Model (직접 로드, HF pipeline 사용 X)
    # =========================================================================
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # =========================================================================
    # 3. Load Dataset
    # =========================================================================
    print(f"Loading {args.dataset} {args.split} set...")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=args.split)
        question_col, answer_col = "question", "answer"
        use_boxed_format = False
    elif args.dataset == "numina":
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

    # =========================================================================
    # 4. Evaluation Loop
    # =========================================================================
    correct = 0
    total = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        question = item[question_col]
        answer_text = item[answer_col]

        if use_boxed_format:
            ground_truth = extract_ground_truth_boxed(answer_text)
        else:
            ground_truth = extract_ground_truth(answer_text)

        if ground_truth is None:
            continue

        # ---------------------------------------------------------------------
        # 4.1 Create prompt (TRM과 동일한 ChatML 형식)
        # ---------------------------------------------------------------------
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # ---------------------------------------------------------------------
        # 4.2 Tokenize (TRM과 동일: 단일 샘플, padding 없음)
        # ---------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # ---------------------------------------------------------------------
        # 4.3 Generate (직접 generate 호출, pipeline 사용 X)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding (TRM과 동일)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # ---------------------------------------------------------------------
        # 4.4 Decode response (TRM과 동일한 방식)
        # ---------------------------------------------------------------------
        # 새로 생성된 토큰만 디코딩 (input_ids 이후)
        response = tokenizer.decode(
            outputs[0, input_ids.size(1):],
            skip_special_tokens=True
        )

        # ---------------------------------------------------------------------
        # 4.5 Extract answer (TRM과 동일한 함수 사용)
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 4.6 Verbose output (TRM과 동일한 형식)
        # ---------------------------------------------------------------------
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

        # Progress update
        if total % 50 == 0:
            print(f"Progress: {total}/{len(dataset)}, Acc: {correct/total*100:.1f}%")

    # =========================================================================
    # 5. Final Results
    # =========================================================================
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 50)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
