"""
Main Training Script for GP2Vec

This module provides the main entry point for training GP2Vec models using
PyTorch Lightning and Hydra for configuration management.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from .module import GP2VecModule
from ..utils.io import S3Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The dataloader.*")


@rank_zero_only
def print_config(config: DictConfig) -> None:
    """Print configuration in a readable format."""
    print("=" * 80)
    print("GP2Vec Training Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(config, resolve=True))
    print("=" * 80)


def create_callbacks(config: DictConfig) -> List[Any]:
    """Create training callbacks based on configuration."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config.get('dirpath', 'checkpoints'),
        filename=checkpoint_config.get('filename', 'gp2vec-{epoch:02d}-{val_loss:.2f}'),
        monitor=checkpoint_config.get('monitor', 'val/loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        auto_insert_metric_name=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_config = config.get('early_stopping', {})
    if early_stopping_config.get('enable', False):
        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val/loss'),
            mode=early_stopping_config.get('mode', 'min'),
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False,
    )
    callbacks.append(lr_monitor)
    
    # Progress bar
    if config.get('progress_bar', {}).get('enable', True):
        progress_bar = RichProgressBar(
            leave=True,
            refresh_rate=config.get('progress_bar', {}).get('refresh_rate', 1),
        )
        callbacks.append(progress_bar)
    
    return callbacks


def create_loggers(config: DictConfig) -> List[Any]:
    """Create loggers based on configuration."""
    loggers = []
    
    # TensorBoard logger
    tb_config = config.get('tensorboard', {})
    if tb_config.get('enable', True):
        tb_logger = TensorBoardLogger(
            save_dir=tb_config.get('save_dir', 'logs'),
            name=tb_config.get('name', 'gp2vec'),
            version=tb_config.get('version', None),
            log_graph=tb_config.get('log_graph', False),
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enable', False):
        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'gp2vec'),
            name=wandb_config.get('name', None),
            tags=wandb_config.get('tags', []),
            log_model=wandb_config.get('log_model', False),
        )
        loggers.append(wandb_logger)
    
    return loggers


def create_strategy(config: DictConfig) -> Optional[Any]:
    """Create distributed training strategy."""
    strategy_config = config.get('strategy', {})
    strategy_type = strategy_config.get('type', 'auto')
    
    if strategy_type == 'ddp':
        return DDPStrategy(
            find_unused_parameters=strategy_config.get('find_unused_parameters', False),
            gradient_as_bucket_view=strategy_config.get('gradient_as_bucket_view', True),
        )
    elif strategy_type == 'fsdp':
        return FSDPStrategy(
            auto_wrap_policy=strategy_config.get('auto_wrap_policy', None),
            mixed_precision=strategy_config.get('mixed_precision', None),
        )
    else:
        return strategy_type  # 'auto', 'dp', etc.


def setup_training_environment(config: DictConfig) -> None:
    """Set up training environment and optimizations."""
    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
        torch.backends.cudnn.deterministic = config.get('cudnn_deterministic', False)
    
    # Set number of threads
    num_threads = config.get('num_threads', None)
    if num_threads:
        torch.set_num_threads(num_threads)
    
    rank_zero_info(f"Training environment configured (seed={seed})")


def validate_config(config: DictConfig) -> None:
    """Validate training configuration."""
    required_fields = ['model', 'data', 'trainer']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Validate data paths
    data_config = config.data
    if 'manifest_path' in data_config and not Path(data_config.manifest_path).exists():
        raise FileNotFoundError(f"Manifest file not found: {data_config.manifest_path}")
    
    if 'metadata_path' in data_config and not Path(data_config.metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {data_config.metadata_path}")
    
    rank_zero_info("Configuration validation passed")


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(config: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Hydra configuration object
    """
    # Print configuration
    print_config(config)
    
    # Validate configuration
    validate_config(config)
    
    # Setup training environment
    setup_training_environment(config)
    
    # Create Lightning module
    rank_zero_info("Creating GP2Vec Lightning module...")
    
    # Merge all config sections into module config
    module_config = {
        **config.get('model', {}),
        **config.get('data', {}),
        **config.get('optimization', {}),
        **config.get('training', {}),
    }
    
    lightning_module = GP2VecModule(**module_config)
    
    # Create callbacks and loggers
    callbacks = create_callbacks(config)
    loggers = create_loggers(config)
    
    # Create distributed strategy
    strategy = create_strategy(config)
    
    # Create trainer
    trainer_config = config.get('trainer', {})
    
    trainer = Trainer(
        # Training configuration
        max_epochs=trainer_config.get('max_epochs', 100),
        max_steps=trainer_config.get('max_steps', -1),
        min_epochs=trainer_config.get('min_epochs', 1),
        
        # Validation configuration
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        
        # Logging configuration
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        
        # Precision configuration
        precision=trainer_config.get('precision', '32-true'),
        
        # Gradient configuration
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        gradient_clip_algorithm=trainer_config.get('gradient_clip_algorithm', 'norm'),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        
        # Device configuration
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        num_nodes=trainer_config.get('num_nodes', 1),
        
        # Distributed configuration
        strategy=strategy,
        
        # Callbacks and logging
        callbacks=callbacks,
        logger=loggers,
        
        # Other options
        enable_checkpointing=True,
        enable_progress_bar=config.get('progress_bar', {}).get('enable', True),
        enable_model_summary=trainer_config.get('enable_model_summary', True),
        
        # Performance optimizations
        benchmark=trainer_config.get('benchmark', None),
        deterministic=trainer_config.get('deterministic', False),
        
        # Debugging options
        fast_dev_run=trainer_config.get('fast_dev_run', False),
        overfit_batches=trainer_config.get('overfit_batches', 0.0),
        limit_train_batches=trainer_config.get('limit_train_batches', 1.0),
        limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
    )
    
    # Resume from checkpoint if specified
    ckpt_path = config.get('resume_from_checkpoint', None)
    if ckpt_path and Path(ckpt_path).exists():
        rank_zero_info(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        ckpt_path = None
    
    # Start training
    rank_zero_info("Starting training...")
    
    try:
        trainer.fit(lightning_module, ckpt_path=ckpt_path)
        
        rank_zero_info("Training completed successfully!")
        
        # Final validation
        if trainer_config.get('run_final_validation', True):
            rank_zero_info("Running final validation...")
            trainer.validate(lightning_module, ckpt_path='best')
        
    except KeyboardInterrupt:
        rank_zero_info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        rank_zero_info("Training session finished")


def train_from_config_file(config_path: str, overrides: Optional[List[str]] = None) -> None:
    """
    Train model from configuration file.
    
    Args:
        config_path: Path to configuration file
        overrides: List of configuration overrides
    """
    # This function allows training without Hydra decorators
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    config_dir = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem
    
    GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_name, overrides=overrides or [])
        main(config)


if __name__ == "__main__":
    main()