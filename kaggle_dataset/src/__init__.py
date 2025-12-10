"""
MathLLM: Qwen-TRM Integrated Model for Mathematical Reasoning
"""

from .config import TRMConfig
from .interface import TRMInterface
from .layers import SwiGLU, TRMAttention, TRMBlock, TRMTransformerBlock, RotaryEmbedding
from .engine import TinyRecursiveTransformer
from .heads import TRMHeads
from .model import QwenTRM
from .train import Trainer, TrainingConfig, EMA
from .dataset import GSM8KDataset, load_gsm8k, create_dataloaders

__all__ = [
    # Config
    "TRMConfig",
    "TrainingConfig",
    # Components
    "TRMInterface",
    "SwiGLU",
    "TRMAttention",
    "TRMBlock",
    "TRMTransformerBlock",  # Backward compatibility alias
    "RotaryEmbedding",
    "TinyRecursiveTransformer",
    "TRMHeads",
    # Full Model
    "QwenTRM",
    # Training
    "Trainer",
    "EMA",
    # Dataset
    "GSM8KDataset",
    "load_gsm8k",
    "create_dataloaders",
]

__version__ = "0.1.0"
