import argparse
import sys
from pathlib import Path
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_splits, get_class_distribution
from src.models.cnn_1d import CNN1DModel
from src.models.imbalance import FocalLoss, get_class_weights
from src.evaluation.metrics import compute_all_metrics, print_metrics, save_metrics
from src.evaluation.latency import comprehensive_latency_benchmark


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=512):
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    
    model.model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    
    model.model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model.model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    return avg_loss, accuracy, all_preds, all_targets, all_probs


def train_cnn(args, X_train, y_train, X_val, y_val, X_test, y_test):
    
    print("\n" + "=" * 70)
    print("Training 1D-CNN")
    print("=" * 70)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"\nModel configuration:")
    print(f"  Input features: {n_features}")
    print(f"  Output classes: {n_classes}")
    print(f"  Conv blocks: {args.n_blocks}")
    print(f"  Filters: {args.filters}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Pooling: {args.pooling}")
    
    # Initialize model
    model = CNN1DModel(
        n_features=n_features,
        n_classes=n_classes,
        n_blocks=args.n_blocks,
        filters=args.filters,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        pooling=args.pooling,
        device=device
    )
    
    # Loss function
    if args.imbalance == 'focal':
        print("\nUsing Focal Loss")
        class_weights = get_class_weights(y_train, method='balanced', return_tensor=True)
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    elif args.imbalance == 'class_weight':
        print("\nUsing Weighted Cross Entropy")
        class_weights = get_class_weights(y_train, method='balanced', return_tensor=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\nUsing standard Cross Entropy")
        criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets, val_probs = evaluate(
            model, val_loader, criterion, device
        )
        
        # Compute F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            model_dir = Path(args.output_dir) / args.dataset / 'cnn_1d'
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir)
            
            print(f"  → New best model saved! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    print(f"\nBest validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Load best model
    model_dir = Path(args.output_dir) / args.dataset / 'cnn_1d'
    model.load(model_dir, device=device)
    
    # Save training history
    with open(model_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    # Validation set
    _, _, val_preds, val_targets, val_probs = evaluate(
        model, val_loader, criterion, device
    )
    val_metrics = compute_all_metrics(val_targets, val_preds, val_probs)
    print_metrics(val_metrics, "1D-CNN - Validation Set")
    save_metrics(val_metrics, model_dir / 'val_metrics.json')
    
    # Test set
    _, _, test_preds, test_targets, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    test_metrics = compute_all_metrics(test_targets, test_preds, test_probs)
    print_metrics(test_metrics, "1D-CNN - Test Set")
    save_metrics(test_metrics, model_dir / 'test_metrics.json')
    
    # Latency benchmark
    if args.benchmark_latency:
        print("\n" + "=" * 70)
        print("Latency Benchmark")
        print("=" * 70)
        latency_results = comprehensive_latency_benchmark(
            model,
            X_test,
            model_path=model_dir,
            model_name=f"1D-CNN ({args.dataset})",
            save_path=model_dir / 'latency.json'
        )
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN for NIDS")
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='cicids2017',
        choices=['cicids2017', 'unsw-nb15'],
        help='Which dataset to use'
    )
    
    # Model architecture
    parser.add_argument('--n-blocks', type=int, default=3, help='Number of conv blocks')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg', 'both'])
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Imbalance handling
    parser.add_argument(
        '--imbalance',
        type=str,
        default='focal',
        choices=['none', 'class_weight', 'focal'],
        help='Imbalance handling strategy'
    )
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for models and results'
    )
    
    # Evaluation
    parser.add_argument(
        '--benchmark-latency',
        action='store_true',
        help='Run latency benchmark'
    )
    
    # Random seed
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # Load data
    print("\n" + "=" * 70)
    print(f"Loading {args.dataset} dataset...")
    print("=" * 70)
    
    splits = load_splits(args.dataset)
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']
    
    # Show class distribution
    print("\nTraining set class distribution:")
    get_class_distribution(y_train)
    
    # Train model
    start_time = time.time()
    model, test_metrics = train_cnn(
        args, X_train, y_train, X_val, y_val, X_test, y_test
    )
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete! Total time: {elapsed_time/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
