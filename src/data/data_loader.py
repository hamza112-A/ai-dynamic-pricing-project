"""
Data loading utilities for experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def load_splits(dataset_name, data_dir="data/splits"):
    """
    Load preprocessed train/val/test splits.
    
    Args:
        dataset_name: 'cicids2017' or 'unsw-nb15'
        data_dir: Directory containing the splits
    
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
    """
    data_dir = Path(data_dir) / dataset_name
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Splits not found in {data_dir}. "
            f"Please run preprocessing first."
        )
    
    print(f"Loading splits from {data_dir}...")
    
    splits = {
        'X_train': np.load(data_dir / 'X_train.npy'),
        'X_val': np.load(data_dir / 'X_val.npy'),
        'X_test': np.load(data_dir / 'X_test.npy'),
        'y_train': np.load(data_dir / 'y_train.npy'),
        'y_val': np.load(data_dir / 'y_val.npy'),
        'y_test': np.load(data_dir / 'y_test.npy'),
    }
    
    print(f"  Train: {splits['X_train'].shape}")
    print(f"  Val:   {splits['X_val'].shape}")
    print(f"  Test:  {splits['X_test'].shape}")
    
    return splits


class NIDSDataset(Dataset):
    """PyTorch Dataset for NIDS data."""
    
    def __init__(self, X, y):
        """
        Args:
            X: Feature array (numpy)
            y: Label array (numpy)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(splits, batch_size=512, num_workers=0):
    """
    Create PyTorch DataLoaders for train/val/test.
    
    Args:
        splits: Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = NIDSDataset(splits['X_train'], splits['y_train'])
    val_dataset = NIDSDataset(splits['X_val'], splits['y_val'])
    test_dataset = NIDSDataset(splits['X_test'], splits['y_test'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    return train_loader, val_loader, test_loader


def get_class_distribution(y):
    """Get class distribution statistics."""
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("\nClass distribution:")
    total = sum(counts)
    for cls, count in sorted(distribution.items()):
        percentage = 100 * count / total
        print(f"  Class {cls}: {count:,} samples ({percentage:.2f}%)")
    
    return distribution


def compute_class_weights(y, method='balanced'):
    """
    Compute class weights for imbalanced data.
    
    Args:
        y: Label array
        method: 'balanced' or 'inverse_sqrt'
    
    Returns:
        Dictionary mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(y)
    
    if method == 'balanced':
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
    elif method == 'inverse_sqrt':
        # Inverse square root of frequency
        unique, counts = np.unique(y, return_counts=True)
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * len(unique)  # Normalize
    else:
        raise ValueError(f"Unknown method: {method}")
    
    class_weights = dict(zip(unique_classes, weights))
    
    print(f"\nClass weights ({method}):")
    for cls, weight in sorted(class_weights.items()):
        print(f"  Class {cls}: {weight:.4f}")
    
    return class_weights


if __name__ == "__main__":
    # Test loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cicids2017")
    args = parser.parse_args()
    
    # Load and inspect
    splits = load_splits(args.dataset)
    
    # Show class distribution
    print("\n" + "=" * 50)
    print("TRAIN SET")
    print("=" * 50)
    train_dist = get_class_distribution(splits['y_train'])
    
    print("\n" + "=" * 50)
    print("VALIDATION SET")
    print("=" * 50)
    val_dist = get_class_distribution(splits['y_val'])
    
    print("\n" + "=" * 50)
    print("TEST SET")
    print("=" * 50)
    test_dist = get_class_distribution(splits['y_test'])
    
    # Compute class weights
    print("\n" + "=" * 50)
    class_weights = compute_class_weights(splits['y_train'], method='balanced')
