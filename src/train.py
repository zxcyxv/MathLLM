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
from tqdm import tqdm

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
    gradient_accumulation_steps: int = 1  # Effective batch = batch_size * accumulation

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./checkpoints"

    # Logging
    log_steps: int = 10

    # Mixed Precision (AMP)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"


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

        # Mixed Precision (AMP)
        self.use_amp = self.config.use_amp
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
            # bfloat16 doesn't need GradScaler, float16 does
            self.scaler = torch.amp.GradScaler('cuda') if self.config.amp_dtype == "float16" else None
            print(f"[Trainer] AMP enabled with {self.config.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            print("[Trainer] AMP disabled, using float32")

    def _setup_optimizer(self) -> AdamW:
        """Setup AdamW optimizer for TRM parameters only (Paper: β1=0.9, β2=0.95)"""
        # Only optimize non-backbone parameters
        self.trm_params = []  # Cache for gradient clipping
        for name, param in self.model.named_parameters():
            if 'backbone' not in name and param.requires_grad:
                self.trm_params.append(param)

        return AdamW(
            self.trm_params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

    def _setup_scheduler(self) -> CosineAnnealingLR:
        """Setup cosine annealing scheduler"""
        # Account for N_sup optimizer steps per accumulated batch
        N_sup = self.model_config.active_N_supervision
        acc_steps = self.config.gradient_accumulation_steps
        # With accumulation, we update once per acc_steps batches, N_sup times
        num_accumulated_batches = len(self.train_dataloader) // acc_steps
        total_steps = num_accumulated_batches * self.config.num_epochs * N_sup
        return CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )

    def train(self):
        """
        Main training loop with Step-wise Gradient Accumulation.

        Key insight: In TRM's Deep Supervision, we must accumulate gradients
        WITHIN each supervision step, not across steps. This ensures:
        1. All micro-batches in the same step use the same weights
        2. Weights are updated after each supervision step completes
        3. Next step uses the updated weights (as per paper)
        """
        self.model.train()
        acc_steps = self.config.gradient_accumulation_steps
        total_batches = len(self.train_dataloader)

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_updates = 0

            # Progress bar for epoch
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                unit="batch",
                dynamic_ncols=True
            )

            # Fast path: no accumulation (no CPU offloading)
            if acc_steps == 1:
                for batch_idx, batch in enumerate(pbar):
                    loss = self._training_step(batch)
                    epoch_loss += loss
                    num_updates += 1

                    # Update progress bar
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': self.global_step
                    })

                    if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        self._evaluate()

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                    self.global_step += 1
            else:
                # Slow path: gradient accumulation with CPU offloading
                micro_batches = []

                for batch_idx, batch in enumerate(pbar):
                    micro_batches.append(batch)

                    if len(micro_batches) == acc_steps:
                        loss = self._training_step_accumulated(micro_batches)
                        epoch_loss += loss
                        num_updates += 1
                        micro_batches = []

                        # Update progress bar
                        lr = self.scheduler.get_last_lr()[0]
                        pbar.set_postfix({
                            'loss': f'{loss:.4f}',
                            'lr': f'{lr:.2e}',
                            'step': self.global_step
                        })

                        if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                            self._evaluate()

                        if self.global_step % self.config.save_steps == 0:
                            self._save_checkpoint()

                        self.global_step += 1

                # Handle remaining micro-batches
                if micro_batches:
                    loss = self._training_step_accumulated(micro_batches)
                    epoch_loss += loss
                    num_updates += 1
                    self.global_step += 1

            pbar.close()

            avg_loss = epoch_loss / max(num_updates, 1)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - Avg Loss: {avg_loss:.4f}")

        # Save final checkpoint after training completes
        print(f"[Train] Training complete. Saving final checkpoint...")
        self._save_checkpoint()

    def _training_step_accumulated(self, micro_batches: list) -> float:
        """
        Step-wise Gradient Accumulation for TRM Deep Supervision.

        Algorithm:
        1. Pre-encode all micro-batches with backbone (frozen, once)
        2. Initialize y, z states for all samples (stored on CPU)
        3. For each supervision step:
           a. zero_grad
           b. For each micro-batch: forward, backward (accumulate gradients)
           c. optimizer.step (update weights)
           d. Save y, z states back to CPU for next step

        This ensures all micro-batches in the same step use the SAME weights,
        and weights are updated BETWEEN steps (matching paper's intention).
        """
        acc_steps = len(micro_batches)
        N_sup = self.model_config.active_N_supervision
        total_loss = 0.0

        # 1. Pre-encode all micro-batches with backbone
        all_hidden_states = []
        all_labels = []
        all_cos = []
        all_sin = []

        # DIS: Create target generator if needed
        dis_generator = None
        if self.model_config.use_dis:
            from .dis_utils import DISTargetGenerator
            dis_generator = DISTargetGenerator(
                vocab_size=self.model_config.vocab_size,
                N_supervision=N_sup,
                noise_schedule=self.model_config.dis_noise_schedule
            )

        with torch.no_grad():
            for batch in micro_batches:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)

                hidden_states = self.model.encode_backbone(input_ids, attention_mask)

                # Compute RoPE once per micro-batch
                B, S, D = hidden_states.shape
                cos, sin = self.model.rotary_emb(hidden_states, S)

                # Store on CPU to save GPU memory
                all_hidden_states.append(hidden_states.cpu())
                all_labels.append(labels.cpu())
                all_cos.append(cos.cpu())
                all_sin.append(sin.cpu())

        # 2. Generate intermediate targets for DIS
        all_target_lists = []
        if dis_generator is not None:
            for labels in all_labels:
                # Generate targets for this batch
                labels_on_device = labels.to(self.device)
                targets = dis_generator.generate_targets(labels_on_device, self.device)
                # Store as list per batch (on CPU to save memory)
                all_target_lists.append([t.cpu() for t in targets])
        else:
            # Standard TRM: same target for all steps
            for labels in all_labels:
                all_target_lists.append([labels] * N_sup)

        # 3. Initialize y, z states for all micro-batches (on CPU)
        all_y = [None] * acc_steps
        all_z = [None] * acc_steps

        # 4. Deep Supervision Loop
        for sup_step in range(N_sup):
            self.optimizer.zero_grad()
            step_loss = 0.0

            # 3a. Accumulate gradients across all micro-batches
            for i in range(acc_steps):
                # Load data to GPU (non_blocking for async transfer)
                hidden_states = all_hidden_states[i].to(self.device, non_blocking=True)
                cos = all_cos[i].to(self.device, non_blocking=True)
                sin = all_sin[i].to(self.device, non_blocking=True)

                # Load step-specific target for DIS
                step_target = all_target_lists[i][sup_step].to(self.device, non_blocking=True)

                # Load states (None on first step, from CPU otherwise)
                y = all_y[i].to(self.device, non_blocking=True) if all_y[i] is not None else None
                z = all_z[i].to(self.device, non_blocking=True) if all_z[i] is not None else None

                # Forward with AMP
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        labels=step_target,  # Use step-specific target
                        y=y,
                        z=z,
                        hidden_states=hidden_states,
                        cos=cos,
                        sin=sin,
                        step_index=sup_step  # Pass step index for DIS
                    )
                    # Scale loss for accumulation
                    loss = outputs['loss'] / acc_steps

                step_loss += outputs['loss'].item()  # Log unscaled loss

                # Backward (accumulates gradients) with optional GradScaler
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Save states to CPU for next supervision step (non_blocking)
                all_y[i] = outputs['y'].to('cpu', non_blocking=True)
                all_z[i] = outputs['z'].to('cpu', non_blocking=True)

            # 3b. Update weights after all micro-batches processed
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trm_params, self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.trm_params, self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()

            # EMA update
            if self.ema is not None:
                self.ema.update()

            total_loss += step_loss / acc_steps  # Average across micro-batches

        # Return average loss across all supervision steps
        return total_loss / N_sup

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single batch training step (no accumulation).
        Fast path without CPU offloading.
        Supports mixed precision (AMP) for faster training.
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch['labels'].to(self.device)

        # Encode backbone once (frozen)
        with torch.no_grad():
            hidden_states = self.model.encode_backbone(input_ids, attention_mask)

        # Compute RoPE once
        B, S, D = hidden_states.shape
        cos, sin = self.model.rotary_emb(hidden_states, S)

        # Initialize states
        y, z = None, None
        total_loss = 0.0

        # N_sup Deep Supervision Loop
        N_sup = self.model_config.active_N_supervision

        # Generate DIS targets if needed
        if self.model_config.use_dis:
            from .dis_utils import DISTargetGenerator
            dis_generator = DISTargetGenerator(
                vocab_size=self.model_config.vocab_size,
                N_supervision=N_sup,
                noise_schedule=self.model_config.dis_noise_schedule
            )
            targets = dis_generator.generate_targets(labels, self.device)
        else:
            targets = [labels] * N_sup

        for sup_step in range(N_sup):
            self.optimizer.zero_grad()

            # Forward with AMP
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(
                    labels=targets[sup_step],  # Use step-specific target
                    y=y,
                    z=z,
                    hidden_states=hidden_states,
                    cos=cos,
                    sin=sin,
                    step_index=sup_step  # Pass step index for DIS
                )
                loss = outputs['loss']

            total_loss += loss.item()

            # Backward with optional GradScaler (for float16)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trm_params, self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trm_params, self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()

            if self.ema is not None:
                self.ema.update()

            y = outputs['y']
            z = outputs['z']

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

                # Encode backbone once, reuse hidden_states
                hidden_states = self.model.encode_backbone(input_ids, attention_mask)

                # Compute RoPE once and reuse across supervision steps
                B, S, D = hidden_states.shape
                cos, sin = self.model.rotary_emb(hidden_states, S)

                # Run full N_sup supervision for evaluation
                y, z = None, None
                N_sup = self.model_config.active_N_supervision
                for step in range(N_sup):
                    outputs = self.model(
                        labels=labels, y=y, z=z,
                        hidden_states=hidden_states,
                        cos=cos, sin=sin,
                        step_index=step
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
