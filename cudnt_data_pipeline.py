#!/usr/bin/env python3
"""
CUDNT Data Pipeline - Efficient Data Loading and Preprocessing
===============================================================

Production-ready data pipeline utilities for CUDNT ML workflows.
Supports efficient loading, preprocessing, and batching for CPU-based ML.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, Callable
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time
import logging

logger = logging.getLogger(__name__)

class CUDNT_DataPipeline:
    """
    Efficient data pipeline for CUDNT ML workflows.
    Handles loading, preprocessing, and batching with CPU optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data pipeline."""
        self.config = config or self._default_config()
        self.executor = ThreadPoolExecutor(max_workers=self.config['num_workers'])
        self.cache = {}  # Data cache for frequently accessed items

        logger.info(f"🗂️ CUDNT Data Pipeline initialized with {self.config['num_workers']} workers")

    def _default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration."""
        return {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': min(4, mp.cpu_count()),
            'prefetch_factor': 2,
            'cache_size': 1000,
            'memory_limit_mb': 1024,
            'enable_augmentation': False
        }

    # ===============================
    # DATA LOADING UTILITIES
    # ===============================

    def load_csv(self, filepath: str, target_column: Optional[str] = None,
                **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load CSV data efficiently."""
        logger.info(f"Loading CSV: {filepath}")

        # Use pandas for efficient loading
        df = pd.read_csv(filepath, **kwargs)

        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column]).values.astype(np.float32)
            y = df[target_column].values.astype(np.float32)
            return X, y
        else:
            return df.values.astype(np.float32), None

    def load_numpy(self, filepath: str) -> np.ndarray:
        """Load numpy array with memory mapping for large files."""
        logger.info(f"Loading numpy array: {filepath}")

        # Use memory mapping for large files
        file_size = os.path.getsize(filepath)
        if file_size > 100 * 1024 * 1024:  # > 100MB
            return np.load(filepath, mmap_mode='r')
        else:
            return np.load(filepath)

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON data."""
        logger.info(f"Loading JSON: {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)

    def create_synthetic_dataset(self, n_samples: int, n_features: int,
                               noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic regression dataset."""
        logger.info(f"Creating synthetic dataset: {n_samples} samples, {n_features} features")

        # Generate features
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Generate target with some pattern
        weights = np.random.randn(n_features).astype(np.float32)
        y_clean = X @ weights + 0.5
        y = y_clean + noise * np.random.randn(n_samples)

        return X.astype(np.float32), y.astype(np.float32)

    # ===============================
    # DATA PREPROCESSING
    # ===============================

    def normalize_data(self, X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Normalize data with different methods."""
        if method == 'standard':
            # Standard scaling: (X - mean) / std
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            X_normalized = (X - mean) / std
            params = {'mean': mean, 'std': std}

        elif method == 'minmax':
            # Min-max scaling: (X - min) / (max - min)
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals == 0, 1, range_vals)
            X_normalized = (X - min_vals) / range_vals
            params = {'min': min_vals, 'max': max_vals}

        elif method == 'robust':
            # Robust scaling: (X - median) / IQR
            median = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1, iqr)
            X_normalized = (X - median) / iqr
            params = {'median': median, 'iqr': iqr}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return X_normalized.astype(np.float32), params

    def apply_normalization(self, X: np.ndarray, params: Dict[str, np.ndarray],
                          method: str = 'standard') -> np.ndarray:
        """Apply normalization parameters to new data."""
        if method == 'standard':
            return ((X - params['mean']) / params['std']).astype(np.float32)
        elif method == 'minmax':
            return ((X - params['min']) / (params['max'] - params['min'])).astype(np.float32)
        elif method == 'robust':
            return ((X - params['median']) / params['iqr']).astype(np.float32)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def augment_data(self, X: np.ndarray, augmentation_config: Dict[str, Any]) -> np.ndarray:
        """Apply data augmentation."""
        if not self.config['enable_augmentation']:
            return X

        augmented_data = []

        for sample in X:
            augmented_sample = sample.copy()

            # Add noise
            if augmentation_config.get('noise', 0) > 0:
                noise = np.random.normal(0, augmentation_config['noise'], sample.shape)
                augmented_sample += noise

            # Random scaling
            if 'scale_range' in augmentation_config:
                scale_factor = np.random.uniform(*augmentation_config['scale_range'])
                augmented_sample *= scale_factor

            augmented_data.append(augmented_sample)

        return np.array(augmented_data, dtype=np.float32)

    # ===============================
    # DATASET CLASS
    # ===============================

    class Dataset:
        """Memory-efficient dataset class."""

        def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    transform: Optional[Callable] = None):
            self.X = X
            self.y = y
            self.transform = transform
            self.indices = np.arange(len(X))

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: Union[int, List[int]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            if isinstance(idx, list):
                X_batch = self.X[idx]
                y_batch = self.y[idx] if self.y is not None else None
            else:
                X_batch = self.X[idx:idx+1]
                y_batch = self.y[idx:idx+1] if self.y is not None else None

            if self.transform:
                X_batch = self.transform(X_batch)

            return X_batch, y_batch

        def shuffle(self):
            """Shuffle dataset indices."""
            np.random.shuffle(self.indices)
            self.X = self.X[self.indices]
            if self.y is not None:
                self.y = self.y[self.indices]

    # ===============================
    # DATA LOADER
    # ===============================

    class DataLoader:
        """Efficient data loader with batching and prefetching."""

        def __init__(self, dataset: 'CUDNT_DataPipeline.Dataset', batch_size: int = 32,
                    shuffle: bool = True, num_workers: int = 0):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.current_idx = 0

            if shuffle:
                self.dataset.shuffle()

        def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
            self.current_idx = 0
            if self.shuffle:
                self.dataset.shuffle()
            return self

        def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            if self.current_idx >= len(self.dataset):
                raise StopIteration

            end_idx = min(self.current_idx + self.batch_size, len(self.dataset))
            batch_indices = list(range(self.current_idx, end_idx))

            X_batch, y_batch = self.dataset[batch_indices]
            self.current_idx = end_idx

            return X_batch, y_batch

        def __len__(self) -> int:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    # ===============================
    # PIPELINE METHODS
    # ===============================

    def create_pipeline(self, data_source: Union[str, Tuple[np.ndarray, np.ndarray]],
                       preprocessing_steps: List[Dict[str, Any]] = None,
                       batch_size: int = None) -> 'DataLoader':
        """Create complete data pipeline."""
        batch_size = batch_size or self.config['batch_size']

        # Load data
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                X, y = self.load_csv(data_source)
            elif data_source.endswith('.npy'):
                X = self.load_numpy(data_source)
                y = None
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            X, y = data_source

        # Apply preprocessing
        if preprocessing_steps:
            for step in preprocessing_steps:
                step_type = step['type']

                if step_type == 'normalize':
                    X, norm_params = self.normalize_data(X, method=step.get('method', 'standard'))
                    # Store params for later use
                    self.cache[f'norm_params_{hash(str(step))}'] = norm_params

                elif step_type == 'augment':
                    X = self.augment_data(X, step.get('config', {}))

        # Create dataset and dataloader
        dataset = self.Dataset(X, y)
        dataloader = self.DataLoader(dataset, batch_size=batch_size,
                                   shuffle=self.config['shuffle'],
                                   num_workers=self.config['num_workers'])

        return dataloader

    def create_train_val_split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                             train_ratio: float = 0.8, shuffle: bool = True) -> Tuple['Dataset', 'Dataset']:
        """Create train/validation split."""
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = self.Dataset(X[train_indices], y[train_indices] if y is not None else None)
        val_dataset = self.Dataset(X[val_indices], y[val_indices] if y is not None else None)

        return train_dataset, val_dataset

    # ===============================
    # UTILITY METHODS
    # ===============================

    def get_data_info(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get dataset information."""
        info = {
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'data_type': X.dtype,
            'memory_usage_mb': X.nbytes / (1024 * 1024),
            'has_target': y is not None
        }

        if y is not None:
            info.update({
                'target_shape': y.shape,
                'target_type': y.dtype,
                'target_memory_mb': y.nbytes / (1024 * 1024)
            })

        return info

    def save_pipeline_config(self, filepath: str):
        """Save pipeline configuration."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_pipeline_config(self, filepath: str):
        """Load pipeline configuration."""
        with open(filepath, 'r') as f:
            self.config.update(json.load(f))

# ===============================
# UTILITY FUNCTIONS
# ===============================

def create_cudnt_data_pipeline(config: Optional[Dict[str, Any]] = None) -> CUDNT_DataPipeline:
    """Create CUDNT data pipeline instance."""
    return CUDNT_DataPipeline(config)

def quick_data_loader(X: np.ndarray, y: Optional[np.ndarray] = None,
                     batch_size: int = 32) -> 'CUDNT_DataPipeline.DataLoader':
    """Quick data loader for simple use cases."""
    pipeline = CUDNT_DataPipeline()
    dataset = pipeline.Dataset(X, y)
    return pipeline.DataLoader(dataset, batch_size=batch_size)

# ===============================
# EXAMPLE USAGE
# ===============================

if __name__ == '__main__':
    # Example usage
    print("🗂️ CUDNT Data Pipeline Demo")
    print("=" * 30)

    # Create pipeline
    pipeline = create_cudnt_data_pipeline()

    # Create synthetic data
    X, y = pipeline.create_synthetic_dataset(1000, 10)
    print(f"Created dataset: {X.shape}, target: {y.shape}")

    # Create preprocessing pipeline
    preprocessing = [
        {'type': 'normalize', 'method': 'standard'}
    ]

    # Create data loader
    dataloader = pipeline.create_pipeline((X, y), preprocessing_steps=preprocessing)

    # Test iteration
    print("Testing data loading...")
    for i, (X_batch, y_batch) in enumerate(dataloader):
        print(f"Batch {i}: X shape {X_batch.shape}, y shape {y_batch.shape}")
        if i >= 2:  # Show first 3 batches
            break

    print("✅ Data pipeline working correctly!")
