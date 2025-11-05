"""
PyTorch Lightning Module for GP2Vec Training

This module implements the LightningModule for training GP2Vec models with
support for distributed training, mixed precision, EMA, and comprehensive logging.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_info
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR

from ..models.gp2vec import GP2Vec, create_gp2vec_model
from ..data.datapipes import create_train_dataloader

logger = logging.getLogger(__name__)


class EMACallback:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, decay: float = 0.999):
        """
        Initialize EMA callback.
        
        Args:
            decay: EMA decay factor
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    
    def register(self, model: nn.Module):
        """Register model parameters for EMA."""
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class GP2VecModule(LightningModule):
    """
    PyTorch Lightning module for GP2Vec training.
    
    Handles training, validation, optimization, and logging for the GP2Vec model.
    """
    
    def __init__(
        self,
        # Model config
        model_config: Optional[Dict] = None,
        model_size: str = "base",
        
        # Data config
        manifest_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        dataloader_config: Optional[Dict] = None,
        
        # Optimization config
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        warmup_steps: int = 10000,
        max_steps: Optional[int] = None,
        
        # Training config
        gradient_clip_val: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        
        # Logging config
        log_every_n_steps: int = 50,
        save_top_k: int = 3,
        
        **kwargs
    ):
        """
        Initialize GP2Vec Lightning module.
        
        Args:
            model_config: Model configuration dictionary
            model_size: Model size preset ('small', 'base', 'large')
            manifest_path: Path to data manifest
            metadata_path: Path to metadata file
            dataloader_config: DataLoader configuration
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            optimizer_type: Optimizer type
            scheduler_type: Learning rate scheduler type
            warmup_steps: Warmup steps for scheduler
            max_steps: Maximum training steps
            gradient_clip_val: Gradient clipping value
            use_ema: Whether to use EMA
            ema_decay: EMA decay factor
            log_every_n_steps: Logging frequency
            save_top_k: Number of best checkpoints to save
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model configuration
        self.model_config = model_config or {}
        self.model_size = model_size
        
        # Data configuration
        self.manifest_path = manifest_path
        self.metadata_path = metadata_path
        self.dataloader_config = dataloader_config or {}
        
        # Optimization configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Training configuration
        self.gradient_clip_val = gradient_clip_val
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Logging configuration
        self.log_every_n_steps = log_every_n_steps
        
        # Initialize model
        self.model = create_gp2vec_model(
            model_size=model_size,
            **self.model_config
        )
        
        # Initialize EMA
        if self.use_ema:
            self.ema = EMACallback(decay=ema_decay)
            self.ema.register(self.model)
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
        rank_zero_info(f"Initialized GP2Vec model with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        waveforms = batch['waveforms']  # (B, C, T)
        metadata = batch.get('metadata', None)
        
        return self.model(waveforms, metadata=metadata)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(batch)
        loss = outputs['loss']
        metrics = outputs['metrics']
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        for metric_name, metric_value in metrics.items():
            if torch.is_tensor(metric_value) and metric_value.numel() == 1:
                self.log(f'train/{metric_name}', metric_value, on_step=True, on_epoch=True)
        
        # Log learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.learning_rate
            self.log('train/lr', current_lr, on_step=True, on_epoch=False)
        
        # Update EMA
        if self.use_ema:
            self.ema.update(self.model)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Use EMA parameters for validation if available
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        
        try:
            outputs = self(batch)
            loss = outputs['loss']
            metrics = outputs['metrics']
            
            # Log validation metrics
            self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            
            for metric_name, metric_value in metrics.items():
                if torch.is_tensor(metric_value) and metric_value.numel() == 1:
                    self.log(f'val/{metric_name}', metric_value, on_step=False, on_epoch=True)
            
            return loss
        
        finally:
            # Restore original parameters
            if self.use_ema:
                self.ema.restore(self.model)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Group parameters by weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't apply weight decay to biases and layer norms
            if len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        # Create optimizer
        if self.optimizer_type.lower() == 'adamw':
            optimizer = AdamW(
                param_groups, 
                lr=self.learning_rate,
                betas=(0.9, 0.98),
                eps=1e-6,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Create scheduler
        if self.scheduler_type.lower() == 'cosine':
            if self.max_steps is None:
                # Estimate max steps from trainer
                if self.trainer.max_steps > 0:
                    max_steps = self.trainer.max_steps
                elif self.trainer.max_epochs > 0:
                    # Rough estimate - will be updated by trainer
                    max_steps = self.trainer.max_epochs * 1000
                else:
                    max_steps = 100000
            else:
                max_steps = self.max_steps
            
            # Linear warmup + cosine annealing
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    cosine_steps = max_steps - self.warmup_steps
                    cosine_progress = (step - self.warmup_steps) / cosine_steps
                    return 0.5 * (1 + torch.cos(torch.tensor(cosine_progress * torch.pi)))
            
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        elif self.scheduler_type.lower() == 'linear':
            # Linear warmup only
            scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        else:
            # No scheduler
            return optimizer
    
    def configure_gradient_clipping(
        self, 
        optimizer: Optimizer, 
        optimizer_idx: int, 
        gradient_clip_val: Optional[Union[int, float]] = None, 
        gradient_clip_algorithm: Optional[str] = None
    ):
        """Configure gradient clipping."""
        if gradient_clip_val is None:
            gradient_clip_val = self.gradient_clip_val
        
        if gradient_clip_val > 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=gradient_clip_val, 
                gradient_clip_algorithm=gradient_clip_algorithm or "norm"
            )
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log epoch-level metrics
        if self.trainer.is_global_zero:
            epoch = self.current_epoch
            step = self.global_step
            
            rank_zero_info(f"Epoch {epoch} completed. Global step: {step}")
            
            # Log model statistics
            if self.use_ema and hasattr(self.ema, 'shadow'):
                param_norm = torch.stack([
                    param.norm() for param in self.ema.shadow.values()
                ]).mean()
                self.log('train/ema_param_norm', param_norm, on_epoch=True)
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.trainer.is_global_zero:
            val_loss = self.trainer.callback_metrics.get('val/loss', None)
            if val_loss is not None:
                rank_zero_info(f"Validation loss: {val_loss:.4f}")
    
    def train_dataloader(self):
        """Create training dataloader."""
        if self.manifest_path is None or self.metadata_path is None:
            raise ValueError("manifest_path and metadata_path must be provided")
        
        return create_train_dataloader(
            manifest_path=self.manifest_path,
            metadata_path=self.metadata_path,
            **self.dataloader_config
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        # For now, use a subset of training data for validation
        # In practice, you would have separate validation manifest
        val_config = self.dataloader_config.copy()
        val_config['max_files'] = val_config.get('max_files', 100) // 10  # Use 10% for validation
        val_config['shuffle_files'] = False
        
        return create_train_dataloader(
            manifest_path=self.manifest_path,
            metadata_path=self.metadata_path,
            **val_config
        )
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save EMA state in checkpoint."""
        if self.use_ema and hasattr(self.ema, 'shadow'):
            checkpoint['ema_state_dict'] = self.ema.shadow.copy()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema.shadow = checkpoint['ema_state_dict']
            rank_zero_info("Loaded EMA state from checkpoint")


def create_lightning_module(
    config: Dict[str, Any]
) -> GP2VecModule:
    """
    Factory function for creating Lightning module from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GP2VecModule instance
    """
    return GP2VecModule(**config)