"""
Training Loop for QwenTRM
Implements Level 1: Deep Supervision (Paper Figure 3)
N_sup times backward per batch + EMA
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from copy import deepcopy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import TRMConfig
from .model import QwenTRM


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    # Optimization (Paper: AdamW with β1=0.9, β2=0.95)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 10
    warmup_steps: int = 100

    # Batching
    batch_size: int = 4

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./checkpoints"

    # Logging
    log_steps: int = 10


class EMA:
    """
    Exponential Moving Average for model parameters.
    Paper recommends decay=0.999 for training stability.

    Optimized with torch._foreach_* for vectorized operations.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay

        # Store as lists for vectorized operations
        self.param_names = []
        self.shadow_params = []
        self.model_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.shadow_params.append(param.data.clone())
                self.model_params.append(param)

        self.backup_params = []

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with EMA - Vectorized"""
        # shadow = decay * shadow + (1 - decay) * param
        # Using foreach operations for ~2-3x speedup
        model_data = [p.data for p in self.model_params]
        torch._foreach_mul_(self.shadow_params, self.decay)
        torch._foreach_add_(self.shadow_params, model_data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation) - Vectorized"""
        self.backup_params = [p.data.clone() for p in self.model_params]
        for param, shadow in zip(self.model_params, self.shadow_params):
            param.data.copy_(shadow)

    def restore(self):
        """Restore original parameters (after evaluation) - Vectorized"""
        for param, backup in zip(self.model_params, self.backup_params):
            param.data.copy_(backup)
        self.backup_params = []

    def state_dict(self):
        shadow_dict = {name: shadow for name, shadow in zip(self.param_names, self.shadow_params)}
        return {'shadow': shadow_dict, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        shadow_dict = state_dict['shadow']
        for i, name in enumerate(self.param_names):
            if name in shadow_dict:
                self.shadow_params[i].copy_(shadow_dict[name])


class Trainer:
    """
    Trainer for QwenTRM with Deep Supervision (Paper Figure 3).

    Level 1: Deep Supervision
        - N_sup times: forward → loss → backward → optimizer.step → zero_grad
        - Each step: y, z detached and passed to next step
        - EMA update after each optimizer step
    """

    def __init__(
        self,
        model: QwenTRM,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()

        # Model config for N_supervision
        self.model_config = model.config

        # Setup optimizer (only TRM parameters, not backbone)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # EMA (Paper: decay=0.999)
        self.ema = None
        if self.model_config.use_ema:
            self.ema = EMA(model, decay=self.model_config.ema_decay)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Device
        self.device = next(model.parameters()).device

    def _setup_optimizer(self) -> AdamW:
        """Setup AdamW optimizer for TRM parameters only (Paper: β1=0.9, β2=0.95)"""
        # Only optimize non-backbone parameters
        trm_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' not in name and param.requires_grad:
                trm_params.append(param)

        return AdamW(
            trm_params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

    def _setup_scheduler(self) -> CosineAnnealingLR:
        """Setup cosine annealing scheduler"""
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        return CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )

    def train(self):
        """Main training loop"""
        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.train_dataloader):
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    self._log_step(loss)

                # Evaluation
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    self._evaluate()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                self.global_step += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - Avg Loss: {avg_loss:.4f}")

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Level 1: Deep Supervision (Paper Figure 3).

        N_sup times:
            - Forward (model handles T-1 no_grad + 1 grad internally)
            - Compute loss
            - Backward
            - Optimizer step
            - Zero grad
            - EMA update
            - Pass detached y, z to next step
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch['labels'].to(self.device)

        # Initialize states
        y, z = None, None
        total_loss = 0.0

        # N_sup Deep Supervision Loop
        N_sup = self.model_config.N_supervision  # 16

        for sup_step in range(N_sup):
            # Forward (internally does T-1 no_grad + 1 grad)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                y=y,
                z=z
            )

            loss = outputs['loss']
            total_loss += loss.item()

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # EMA update
            if self.ema is not None:
                self.ema.update()

            # Get detached states for next supervision step
            y = outputs['y']
            z = outputs['z']

        # Return average loss across supervision steps
        return total_loss / N_sup

    def _evaluate(self):
        """Run evaluation with EMA parameters"""
        self.model.eval()

        # Apply EMA parameters for evaluation
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)

                # Run full N_sup supervision for evaluation
                y, z = None, None
                for _ in range(self.model_config.N_supervision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        y=y, z=z
                    )
                    y = outputs['y']
                    z = outputs['z']

                total_loss += outputs['loss'].item()
                num_batches += 1

        # Restore original parameters
        if self.ema is not None:
            self.ema.restore()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[Eval] Step {self.global_step} - Loss: {avg_loss:.4f}")

        self.model.train()

    def _log_step(self, loss: float):
        """Log training progress"""
        lr = self.scheduler.get_last_lr()[0]
        print(f"[Train] Step {self.global_step} - Loss: {loss:.4f} - LR: {lr:.2e}")

    def _save_checkpoint(self):
        """Save model checkpoint"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save TRM components only (not backbone)
        save_dict = {
            'interface': self.model.interface.state_dict(),
            'engine': self.model.engine.state_dict(),
            'heads': self.model.heads.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.model.config,
        }

        # Save EMA if used
        if self.ema is not None:
            save_dict['ema'] = self.ema.state_dict()

        torch.save(save_dict, os.path.join(checkpoint_path, "trm_model.pt"))

        print(f"[Save] Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        state = torch.load(
            os.path.join(checkpoint_path, "trm_model.pt"),
            map_location=self.device
        )

        self.model.interface.load_state_dict(state['interface'])
        self.model.engine.load_state_dict(state['engine'])
        self.model.heads.load_state_dict(state['heads'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state['global_step']
        self.epoch = state['epoch']

        # Load EMA if present
        if self.ema is not None and 'ema' in state:
            self.ema.load_state_dict(state['ema'])

        print(f"[Load] Checkpoint loaded from {checkpoint_path}")
