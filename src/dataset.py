"""
Dataset classes for math reasoning tasks
"""

import re
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset


class GSM8KDataset(Dataset):
    """
    GSM8K Dataset for math word problems.

    Expected format from HuggingFace 'gsm8k':
        - question: str
        - answer: str (contains step-by-step solution ending with "#### <number>")
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ):
        """
        Args:
            data: List of {"question": ..., "answer": ...}
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Optional custom prompt template
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.prompt_template = prompt_template or (
            "Question: {question}\n"
            "Answer: {answer}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        question = item["question"]
        answer = item["answer"]

        # Extract final numeric answer from "#### <number>"
        final_answer = self._extract_final_answer(answer)

        # Format prompt
        full_text = self.prompt_template.format(
            question=question,
            answer=answer
        )

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels = input_ids (for causal LM)
        # Set padding tokens to -100 to ignore in loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "final_answer": final_answer
        }

    def _extract_final_answer(self, answer: str) -> int:
        """Extract numeric answer from '#### <number>' format"""
        match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer)
        if match:
            # Remove commas and convert to int
            num_str = match.group(1).replace(",", "")
            try:
                return int(float(num_str))
            except ValueError:
                return 0
        return 0


def load_gsm8k(
    tokenizer,
    max_length: int = 512,
    split: str = "train",
    subset_size: Optional[int] = None
) -> GSM8KDataset:
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        split: "train" or "test"
        subset_size: Optional limit on dataset size (for quick testing)

    Returns:
        GSM8KDataset instance
    """
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split=split)

    # Convert to list of dicts
    data = [{"question": item["question"], "answer": item["answer"]}
            for item in dataset]

    # Optional subset for quick testing
    if subset_size is not None:
        data = data[:subset_size]

    return GSM8KDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length
    )


def create_dataloaders(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    num_workers: int = 0
):
    """
    Create train and test dataloaders for GSM8K.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        train_subset: Optional limit on train size
        test_subset: Optional limit on test size
        num_workers: DataLoader workers

    Returns:
        (train_dataloader, test_dataloader)
    """
    from torch.utils.data import DataLoader

    train_dataset = load_gsm8k(
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        subset_size=train_subset
    )

    test_dataset = load_gsm8k(
        tokenizer=tokenizer,
        max_length=max_length,
        split="test",
        subset_size=test_subset
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
