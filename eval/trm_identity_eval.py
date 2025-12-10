"""
TRM Evaluation Script
Supports both:
- Identity test (untrained TRM)
- Trained TRM evaluation (with checkpoint loading)
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


def load_trm_checkpoint(model: QwenTRM, checkpoint_path: str, device: str = "cuda"):
    """Load trained TRM weights from checkpoint."""
    ckpt_file = os.path.join(checkpoint_path, "trm_model.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"Loading checkpoint from: {ckpt_file}")
    state = torch.load(ckpt_file, map_location=device, weights_only=False)

    model.interface.load_state_dict(state['interface'])
    model.engine.load_state_dict(state['engine'])
    model.heads.load_state_dict(state['heads'])

    print(f"[Load] Checkpoint loaded (step={state.get('global_step', '?')}, epoch={state.get('epoch', '?')})")
    return model


def evaluate_trm(
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
    checkpoint_path: str = None,
    num_samples: int = None,
    max_new_tokens: int = 512,
    num_supervision_steps: int = 16,
    batch_size: int = 1,
    use_amp: bool = True,
    device: str = "cuda"
):
    """Evaluate QwenTRM on GSM8K test set.

    Args:
        model_name: Backbone model name
        checkpoint_path: Path to trained checkpoint (None for identity test)
        num_samples: Number of samples to evaluate (None for all)
        max_new_tokens: Max tokens to generate
        num_supervision_steps: Number of supervision steps (1 for identity, 16 for trained)
        batch_size: Batch size for evaluation
        use_amp: Use automatic mixed precision (bfloat16)
        device: Device to use
    """
    is_trained = checkpoint_path is not None

    print(f"Loading QwenTRM with backbone: {model_name}")
    print(f"Mode: {'Trained' if is_trained else 'Identity Test'}")
    print(f"Supervision steps: {num_supervision_steps}")
    print(f"Batch size: {batch_size}")
    print(f"AMP: {use_amp}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batch generation

    # Load QwenTRM
    config = TRMConfig()
    model = QwenTRM.from_pretrained_backbone(
        backbone_name=model_name,
        config=config,
        device=device,
        init_lm_head=True
    )

    # Load checkpoint if provided
    if checkpoint_path:
        model = load_trm_checkpoint(model, checkpoint_path, device)

    # Convert TRM components to bfloat16 if AMP enabled
    if use_amp:
        print("Converting TRM components to bfloat16...")
        model.interface = model.interface.to(torch.bfloat16)
        model.engine = model.engine.to(torch.bfloat16)
        model.heads = model.heads.to(torch.bfloat16)

    model.eval()

    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")

    # Prepare all data
    all_items = []
    for item in dataset:
        ground_truth = extract_ground_truth(item["answer"])
        if ground_truth is not None:
            all_items.append({
                "question": item["question"],
                "answer": item["answer"],
                "ground_truth": ground_truth,
                "prompt": format_prompt(item["question"])
            })

    print(f"Valid samples: {len(all_items)}")

    correct = 0
    total = 0
    results = []

    # Process in batches
    num_batches = (len(all_items) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating TRM"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_items))
        batch_items = all_items[start_idx:end_idx]

        # Tokenize batch
        prompts = [item["prompt"] for item in batch_items]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                # Generate using QwenTRM
                generated = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_new_tokens,
                    num_supervision_steps=num_supervision_steps
                )

        # Decode and evaluate each sample in batch
        for i, item in enumerate(batch_items):
            response = tokenizer.decode(generated[i], skip_special_tokens=True)
            prompt_len = len(item["prompt"])
            response_only = response[prompt_len:]

            predicted = extract_answer(response_only)
            ground_truth = item["ground_truth"]

            is_correct = (predicted == ground_truth)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": item["question"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "response": response_only[:500]
            })

        # Progress update
        if (batch_idx + 1) % 10 == 0:
            print(f"Progress: {total}/{len(all_items)}, Accuracy: {correct/total*100:.2f}%")

    accuracy = correct / total * 100

    print("\n" + "=" * 50)
    print(f"Model: QwenTRM (backbone: {model_name})")
    print(f"Mode: {'Trained' if is_trained else 'Identity Test'}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Supervision Steps: {num_supervision_steps}")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

    return {
        "model": f"QwenTRM({model_name})",
        "checkpoint": checkpoint_path,
        "supervision_steps": num_supervision_steps,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


# Keep old function name for backwards compatibility
def evaluate_trm_identity(*args, **kwargs):
    """Backwards compatibility wrapper."""
    return evaluate_trm(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Backbone model name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained TRM checkpoint (e.g., ./checkpoints/trm/checkpoint-0)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--supervision_steps", type=int, default=None,
                        help="Number of supervision steps (default: 1 for identity, 16 for trained)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation (default: 4)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (bfloat16)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable AMP, use float32")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (default: auto-generated)")

    args = parser.parse_args()

    # Set default supervision steps based on mode
    if args.supervision_steps is None:
        args.supervision_steps = 16 if args.checkpoint else 1

    # Set default output file based on mode
    if args.output is None:
        args.output = "trm_trained_results.json" if args.checkpoint else "trm_identity_results.json"

    results = evaluate_trm(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        num_supervision_steps=args.supervision_steps,
        batch_size=args.batch_size,
        use_amp=args.amp
    )

    with open(args.output, "w") as f:
        json.dump({
            "model": results["model"],
            "checkpoint": results["checkpoint"],
            "supervision_steps": results["supervision_steps"],
            "total": results["total"],
            "correct": results["correct"],
            "accuracy": results["accuracy"]
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
