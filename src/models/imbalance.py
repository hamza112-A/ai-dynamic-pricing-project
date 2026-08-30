import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def apply_smote(X_train, y_train, random_state=42, k_neighbors=5, sampling_strategy='auto'):
    
    print("Applying SMOTE...")
    print(f"  Original shape: {X_train.shape}")
    
    # Get class distribution before SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    print("  Class distribution before SMOTE:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count:,} samples")
    
    # Apply SMOTE
    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  Resampled shape: {X_resampled.shape}")
    
    # Get class distribution after SMOTE
    unique, counts = np.unique(y_resampled, return_counts=True)
    print("  Class distribution after SMOTE:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count:,} samples")
    
    return X_resampled, y_resampled


def get_class_weights(y, method='balanced', return_tensor=False):
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    if method == 'balanced':
        # sklearn's balanced weighting
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
    elif method == 'inverse':
        # Inverse frequency
        unique, counts = np.unique(y, return_counts=True)
        weights = len(y) / (n_classes * counts)
    elif method == 'inverse_sqrt':
        # Inverse square root of frequency
        unique, counts = np.unique(y, return_counts=True)
        weights = np.sqrt(len(y) / counts)
        weights = weights / weights.sum() * n_classes
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if return_tensor:
        return torch.FloatTensor(weights)
    else:
        return dict(zip(unique_classes, weights))


class WeightedFocalLoss(nn.Module):
    
    
    def __init__(self, class_weights, gamma=2.0, reduction='mean'):
        
        super().__init__()
        if isinstance(class_weights, list):
            class_weights = torch.FloatTensor(class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma, reduction=reduction)
    
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)


def evaluate_imbalance_strategy(
    model,
    X_train, y_train,
    X_val, y_val,
    strategy='none'
):
    
    print(f"\nEvaluating imbalance strategy: {strategy}")
    print("=" * 70)
    
    if strategy == 'none':
        # No imbalance handling
        print("Training without imbalance handling...")
        model.fit(X_train, y_train)
    
    elif strategy == 'class_weight':
        # Use class weights
        print("Computing class weights...")
        class_weights = get_class_weights(y_train, method='balanced')
        
        # For sklearn models
        if hasattr(model, 'class_weight'):
            model.class_weight = class_weights
        
        model.fit(X_train, y_train)
    
    elif strategy == 'smote':
        # Apply SMOTE
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        model.fit(X_resampled, y_resampled)
    
    elif strategy == 'focal':
        # Focal loss (for neural networks only)
        print("Note: Focal loss requires integration in training loop")
        # This would be handled in the training script
        raise NotImplementedError(
            "Focal loss must be integrated in the training loop. "
            "Use the FocalLoss class in your training script."
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Evaluate on validation set
    from sklearn.metrics import f1_score, accuracy_score
    
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    
    return model, {'accuracy': accuracy, 'macro_f1': macro_f1}


if __name__ == "__main__":
    # Quick test
    print("Testing imbalance handling utilities...")
    
    # Generate imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Heavily imbalanced: 70% class 0, 20% class 1, 10% class 2
    y_train = np.concatenate([
        np.zeros(700),
        np.ones(200),
        np.full(100, 2)
    ]).astype(int)
    
    X_train = np.random.randn(n_samples, n_features)
    
    print("\nOriginal class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(y_train):.1f}%)")
    
    # Test class weights
    print("\n" + "=" * 70)
    print("Class Weights")
    print("=" * 70)
    
    weights_balanced = get_class_weights(y_train, method='balanced')
    print("\nBalanced weights:")
    for cls, weight in sorted(weights_balanced.items()):
        print(f"  Class {cls}: {weight:.4f}")
    
    weights_sqrt = get_class_weights(y_train, method='inverse_sqrt')
    print("\nInverse sqrt weights:")
    for cls, weight in sorted(weights_sqrt.items()):
        print(f"  Class {cls}: {weight:.4f}")
    
    # Test SMOTE
    print("\n" + "=" * 70)
    print("SMOTE")
    print("=" * 70)
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    
    # Test Focal Loss
    print("\n" + "=" * 70)
    print("Focal Loss")
    print("=" * 70)
    
    # Create dummy predictions
    batch_size = 32
    n_classes = 3
    logits = torch.randn(batch_size, n_classes)
    targets = torch.randint(0, n_classes, (batch_size,))
    
    # Standard cross entropy
    ce_loss = F.cross_entropy(logits, targets)
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    
    # Focal loss without weights
    focal = FocalLoss(gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"Focal Loss (gamma=2.0): {focal_loss.item():.4f}")
    
    # Focal loss with weights
    alpha = torch.FloatTensor([1.0, 2.0, 3.0])
    weighted_focal = FocalLoss(alpha=alpha, gamma=2.0)
    weighted_focal_loss = weighted_focal(logits, targets)
    print(f"Weighted Focal Loss: {weighted_focal_loss.item():.4f}")
    
    print("\n✓ Imbalance handling test passed")
