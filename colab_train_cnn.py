"""
Google Colab CNN Training Script
Upload this file and your data to Colab, then run: !python colab_train_cnn.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time

print("="*70)
print("NIDS 1D-CNN Training on Google Colab")
print("="*70)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = '/content/drive/MyDrive/data'
    print(f"Using Google Drive data from: {DATA_PATH}")
except:
    # Fallback for local files
    DATA_PATH = '.'
    print("Using local data files")

# Load data
print("\nLoading data...")
X_train = np.load(f'{DATA_PATH}/X_train.npy').astype(np.float32)
y_train = np.load(f'{DATA_PATH}/y_train.npy').astype(np.int64)
X_val = np.load(f'{DATA_PATH}/X_val.npy').astype(np.float32)
y_val = np.load(f'{DATA_PATH}/y_val.npy').astype(np.int64)
X_test = np.load(f'{DATA_PATH}/X_test.npy').astype(np.float32)
y_test = np.load(f'{DATA_PATH}/y_test.npy').astype(np.int64)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Classes: {len(np.unique(y_train))}")

# CRITICAL: Check data validity
print("\nData validation:")
print(f"X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
print(f"y_train unique: {np.unique(y_train)}")
print(f"Any NaN in X_train: {np.isnan(X_train).any()}")
print(f"Any Inf in X_train: {np.isinf(X_train).any()}")

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nClass distribution:")
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt:,} samples ({cnt/len(y_train)*100:.2f}%)")

# Define 1D-CNN model
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, filters=64, dropout=0.3):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(filters, filters*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(filters*2, filters*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters*4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv_blocks(x)
        x = self.fc(x)
        return x

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

print("\n" + "="*70)
print("Model Setup")
print("="*70)

# Create model
model = CNN1D(X_train.shape[1], len(np.unique(y_train)), filters=64, dropout=0.3).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Calculate class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"\nClass weights computed for {len(class_weights)} classes")

# Loss and optimizer
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

# Data loaders
batch_size = 4096 if torch.cuda.is_available() else 2048
print(f"Batch size: {batch_size}")

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Training loop
num_epochs = 50
best_val_f1 = 0
patience = 10
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

print("\n" + "="*70)
print(f"Training for {num_epochs} epochs")
print("="*70 + "\n")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    y_val_pred = []
    y_val_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
            y_val_pred.extend(predicted.cpu().numpy())
            y_val_true.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_f1 = precision_recall_fscore_support(y_val_true, y_val_pred, average='macro', zero_division=0)[2]
    
    # Update scheduler
    scheduler.step(val_f1)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Early stopping and save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cnn_colab.pt')
        print(f"  → Best model saved! (F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print()

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Training complete in {total_time/60:.1f} minutes")
print(f"Best validation F1: {best_val_f1:.4f}")
print(f"{'='*70}\n")

# Evaluate on test set
print("Evaluating on test set...")
model.load_state_dict(torch.load('best_cnn_colab.pt'))
model.eval()

test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

y_pred_list = []
y_true_list = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(batch_y.numpy())

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

print(f"\n{'='*70}")
print("TEST SET RESULTS")
print(f"{'='*70}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"{'='*70}\n")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'macro_precision': float(macro_precision),
    'macro_recall': float(macro_recall),
    'macro_f1': float(macro_f1),
    'per_class': {},
    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    'training_history': history
}

for i in range(len(precision)):
    metrics['per_class'][f'Class_{i}'] = {
        'precision': float(precision[i]),
        'recall': float(recall[i]),
        'f1': float(f1[i]),
        'support': int(support[i])
    }

with open('cnn_colab_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✓ Results saved to cnn_colab_results.json")
print("✓ Model saved to best_cnn_colab.pt")
print("\nDone!")
