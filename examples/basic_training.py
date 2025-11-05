"""
Basic GP2Vec Training Example

This example shows how to train a GP2Vec model from scratch
using the provided configuration system.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_gp2vec(cfg: DictConfig) -> None:
    """
    Train GP2Vec model with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    logger.info("Starting GP2Vec training example")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Import here to avoid issues if dependencies aren't installed
    from gp2vec.train.train import main as train_main
    
    # Run training
    train_main(cfg)


if __name__ == "__main__":
    train_gp2vec()