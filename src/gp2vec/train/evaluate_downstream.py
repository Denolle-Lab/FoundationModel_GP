"""
Downstream Task Evaluation for GP2Vec

This module provides evaluation tools for testing GP2Vec representations
on downstream tasks such as phase picking, tremor detection, and magnitude estimation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..models.gp2vec import GP2Vec
from ..data.decoder import process_stream_for_training

logger = logging.getLogger(__name__)


class DownstreamDataset(Dataset):
    """Dataset for downstream task evaluation."""
    
    def __init__(
        self,
        waveforms: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Initialize downstream dataset.
        
        Args:
            waveforms: Waveform data (N, C, T)
            labels: Task labels (N,) or (N, num_classes)
            metadata: Optional metadata for each sample
        """
        self.waveforms = waveforms
        self.labels = labels
        self.metadata = metadata or [{}] * len(waveforms)
        
        assert len(self.waveforms) == len(self.labels)
        assert len(self.waveforms) == len(self.metadata)
    
    def __len__(self) -> int:
        return len(self.waveforms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        return self.waveforms[idx], self.labels[idx], self.metadata[idx]


class LinearProbe(nn.Module):
    """Linear probe for evaluating learned representations."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pooling: str = "mean",  # 'mean', 'max', 'cls', 'attention'
        dropout: float = 0.1,
    ):
        """
        Initialize linear probe.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            pooling: Pooling method for sequence features
            dropout: Dropout rate
        """
        super().__init__()
        
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        
        if pooling == "attention":
            self.attention = nn.Linear(input_dim, 1)
        
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear probe.
        
        Args:
            features: Input features (B, T, D) or (B, D)
            
        Returns:
            Class logits (B, num_classes)
        """
        if features.dim() == 3:
            # Pool sequence features
            if self.pooling == "mean":
                pooled = features.mean(dim=1)  # (B, D)
            elif self.pooling == "max":
                pooled = features.max(dim=1)[0]  # (B, D)
            elif self.pooling == "cls":
                pooled = features[:, 0, :]  # (B, D) - use first token
            elif self.pooling == "attention":
                # Attention pooling
                attn_weights = F.softmax(self.attention(features), dim=1)  # (B, T, 1)
                pooled = (features * attn_weights).sum(dim=1)  # (B, D)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
        else:
            pooled = features
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


class DownstreamEvaluator:
    """
    Evaluator for downstream tasks using GP2Vec representations.
    
    Supports both linear probing and fine-tuning evaluation protocols.
    """
    
    def __init__(
        self,
        model: GP2Vec,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        feature_layer: int = -1,
    ):
        """
        Initialize downstream evaluator.
        
        Args:
            model: Trained GP2Vec model
            device: Device for computation
            feature_layer: Which layer to extract features from
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.feature_layer = feature_layer
        
        logger.info(f"Initialized evaluator on {device}")
    
    def extract_features(
        self,
        dataloader: DataLoader,
        use_metadata: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from waveforms using trained model.
        
        Args:
            dataloader: DataLoader with waveform data
            use_metadata: Whether to use metadata conditioning
            
        Returns:
            Tuple of (features, labels)
        """
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                waveforms, labels, metadata = batch
                waveforms = waveforms.to(self.device)
                
                # Extract features
                if use_metadata and any(meta for meta in metadata):
                    # Convert metadata to tensor format
                    metadata_tensors = self._process_metadata_batch(metadata)
                    features = self.model.encode(waveforms, metadata_tensors)
                else:
                    features = self.model.encode(waveforms)
                
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return features, labels
    
    def _process_metadata_batch(self, metadata_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert metadata list to tensor format."""
        # This is a simplified version - in practice, you'd need to handle
        # the full metadata processing pipeline
        if not metadata_list or not any(metadata_list):
            return {}
        
        # Extract common metadata fields
        metadata_tensors = {}
        
        # Example: extract numeric fields
        for key in ['latitude', 'longitude', 'elevation', 'sampling_rate']:
            values = []
            for meta in metadata_list:
                values.append(meta.get(key, 0.0))
            
            if values:
                metadata_tensors[key] = torch.tensor(values, device=self.device)
        
        return metadata_tensors
    
    def linear_probe_evaluation(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_classes: int,
        **probe_kwargs
    ) -> Dict[str, float]:
        """
        Evaluate using linear probe on frozen features.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            test_features: Test features
            test_labels: Test labels
            num_classes: Number of classes
            **probe_kwargs: Arguments for LinearProbe
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get feature dimension
        if train_features.dim() == 3:
            feature_dim = train_features.size(-1)
        else:
            feature_dim = train_features.size(1)
        
        # Create linear probe
        probe = LinearProbe(
            input_dim=feature_dim,
            num_classes=num_classes,
            **probe_kwargs
        ).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(
            train_features.to(self.device),
            train_labels.to(self.device)
        )
        test_dataset = TensorDataset(
            test_features.to(self.device),
            test_labels.to(self.device)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Train probe
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        probe.train()
        for epoch in range(100):  # Simple training loop
            total_loss = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                
                logits = probe(batch_features)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        # Evaluate probe
        probe.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                logits = probe(batch_features)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())
                all_probs.append(probs.cpu())
        
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
        }
        
        # Add AUC if binary classification
        if num_classes == 2:
            metrics['auc'] = roc_auc_score(labels, probs[:, 1])
        
        return metrics
    
    def sklearn_evaluation(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        pooling: str = "mean",
    ) -> Dict[str, float]:
        """
        Evaluate using sklearn logistic regression.
        
        Args:
            train_features: Training features
            train_labels: Training labels  
            test_features: Test features
            test_labels: Test labels
            pooling: Pooling method for sequence features
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Pool features if needed
        if train_features.dim() == 3:
            if pooling == "mean":
                train_X = train_features.mean(dim=1).numpy()
                test_X = test_features.mean(dim=1).numpy()
            elif pooling == "max":
                train_X = train_features.max(dim=1)[0].numpy()
                test_X = test_features.max(dim=1)[0].numpy()
            else:
                raise ValueError(f"Unsupported pooling for sklearn: {pooling}")
        else:
            train_X = train_features.numpy()
            test_X = test_features.numpy()
        
        train_y = train_labels.numpy()
        test_y = test_labels.numpy()
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_X, train_y)
        
        # Predict
        preds = clf.predict(test_X)
        probs = clf.predict_proba(test_X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(test_y, preds),
            'f1_macro': f1_score(test_y, preds, average='macro'),
            'f1_weighted': f1_score(test_y, preds, average='weighted'),
        }
        
        # Add AUC if binary
        if len(np.unique(test_y)) == 2:
            metrics['auc'] = roc_auc_score(test_y, probs[:, 1])
        
        return metrics
    
    def evaluate_phase_picking(
        self,
        waveforms: torch.Tensor,
        pick_labels: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
        test_size: float = 0.3,
    ) -> Dict[str, float]:
        """
        Evaluate on phase picking task.
        
        Args:
            waveforms: Waveform data (N, C, T)
            pick_labels: Binary labels indicating picks (N,)
            metadata: Optional metadata
            test_size: Test set fraction
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating phase picking task...")
        
        # Create dataset and dataloader
        dataset = DownstreamDataset(waveforms, pick_labels, metadata)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract features
        features, labels = self.extract_features(dataloader)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Evaluate with both methods
        linear_metrics = self.linear_probe_evaluation(
            X_train, y_train, X_test, y_test, num_classes=2
        )
        
        sklearn_metrics = self.sklearn_evaluation(
            X_train, y_train, X_test, y_test
        )
        
        return {
            'linear_probe': linear_metrics,
            'sklearn': sklearn_metrics,
        }
    
    def evaluate_tremor_detection(
        self,
        waveforms: torch.Tensor,
        tremor_labels: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
        test_size: float = 0.3,
    ) -> Dict[str, float]:
        """
        Evaluate on tremor detection task.
        
        Args:
            waveforms: Waveform data (N, C, T)
            tremor_labels: Binary labels indicating tremor (N,)
            metadata: Optional metadata
            test_size: Test set fraction
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating tremor detection task...")
        
        # Same as phase picking but different interpretation
        return self.evaluate_phase_picking(
            waveforms, tremor_labels, metadata, test_size
        )
    
    def evaluate_magnitude_estimation(
        self,
        waveforms: torch.Tensor,
        magnitudes: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
        test_size: float = 0.3,
        num_magnitude_bins: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate on magnitude estimation (as classification).
        
        Args:
            waveforms: Waveform data (N, C, T)
            magnitudes: Continuous magnitude values (N,)
            metadata: Optional metadata
            test_size: Test set fraction
            num_magnitude_bins: Number of magnitude bins
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating magnitude estimation task...")
        
        # Bin magnitudes into discrete classes
        mag_bins = torch.linspace(magnitudes.min(), magnitudes.max(), num_magnitude_bins + 1)
        magnitude_labels = torch.bucketize(magnitudes, mag_bins[1:-1])
        
        # Create dataset and dataloader
        dataset = DownstreamDataset(waveforms, magnitude_labels, metadata)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract features
        features, labels = self.extract_features(dataloader)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Evaluate
        linear_metrics = self.linear_probe_evaluation(
            X_train, y_train, X_test, y_test, num_classes=num_magnitude_bins
        )
        
        sklearn_metrics = self.sklearn_evaluation(
            X_train, y_train, X_test, y_test
        )
        
        return {
            'linear_probe': linear_metrics,
            'sklearn': sklearn_metrics,
        }


def load_pretrained_model(checkpoint_path: str, device: str = "auto") -> GP2Vec:
    """
    Load pretrained GP2Vec model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded GP2Vec model
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict and config
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning module)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # Create model (you might need to infer config from checkpoint)
    # This is a simplified version - in practice you'd save the config too
    model = GP2Vec()  # Use default config or load from checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Loaded pretrained model from {checkpoint_path}")
    return model


def run_downstream_evaluation(
    model_path: str,
    data_config: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive downstream evaluation.
    
    Args:
        model_path: Path to pretrained model
        data_config: Configuration for downstream data
        output_path: Path to save results
        
    Returns:
        Evaluation results
    """
    logger.info("Starting downstream evaluation...")
    
    # Load model
    model = load_pretrained_model(model_path)
    evaluator = DownstreamEvaluator(model)
    
    results = {}
    
    # Load and evaluate each task
    for task_name, task_config in data_config.items():
        logger.info(f"Evaluating task: {task_name}")
        
        # Load task data (implementation depends on data format)
        waveforms, labels, metadata = load_task_data(task_config)
        
        if task_name == "phase_picking":
            task_results = evaluator.evaluate_phase_picking(
                waveforms, labels, metadata
            )
        elif task_name == "tremor_detection":
            task_results = evaluator.evaluate_tremor_detection(
                waveforms, labels, metadata
            )
        elif task_name == "magnitude_estimation":
            task_results = evaluator.evaluate_magnitude_estimation(
                waveforms, labels, metadata
            )
        else:
            logger.warning(f"Unknown task: {task_name}")
            continue
        
        results[task_name] = task_results
        
        # Log results
        logger.info(f"Results for {task_name}:")
        for method, metrics in task_results.items():
            for metric, value in metrics.items():
                logger.info(f"  {method}.{metric}: {value:.4f}")
    
    # Save results if path provided
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results


def load_task_data(task_config: Dict) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Load task-specific data.
    
    This is a placeholder - implement based on your data format.
    """
    # This would load your specific downstream task data
    # For now, return dummy data
    n_samples = 1000
    n_channels = 3
    seq_len = 3000
    
    waveforms = torch.randn(n_samples, n_channels, seq_len)
    labels = torch.randint(0, 2, (n_samples,))
    metadata = [{}] * n_samples
    
    return waveforms, labels, metadata


if __name__ == "__main__":
    # Example usage
    model_path = "path/to/pretrained/model.ckpt"
    
    data_config = {
        "phase_picking": {
            "data_path": "path/to/phase_picking_data",
        },
        "tremor_detection": {
            "data_path": "path/to/tremor_data", 
        },
    }
    
    results = run_downstream_evaluation(model_path, data_config)
    print("Evaluation completed:", results)