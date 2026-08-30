"""Simple CNN training with PyTorch."""
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time

print("Loading data...")
X_train = np.load('data/splits/cicids2017/X_train.npy').astype(np.float32)
y_train = np.load('data/splits/cicids2017/y_train.npy').astype(np.int64)
X_val = np.load('data/splits/cicids2017/X_val.npy').astype(np.float32)
y_val = np.load('data/splits/cicids2017/y_val.npy').astype(np.int64)
X_test = np.load('data/splits/cicids2017/X_test.npy').astype(np.float32)
y_test = np.load('data/splits/cicids2017/y_test.npy').astype(np.int64)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Calculate class weights for focal loss
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.FloatTensor(class_weights).to(device)

# Create model
model = SimpleCNN(X_train.shape[1], len(np.unique(y_train))).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders - smaller batch size
batch_size = 2048
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"\nTraining for 20 epochs...")
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(20):
    # Training
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/20 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_cnn_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('best_cnn_model.pt'))

# Evaluate on test set
print("\nEvaluating on test set...")
model.eval()
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
macro_precision = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[0]
macro_recall = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[1]
macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]

print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# Save model
output_dir = Path('models/cicids2017/cnn')
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_dir / 'model.pt')
print(f"\nModel saved to {output_dir}")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'macro_precision': float(macro_precision),
    'macro_recall': float(macro_recall),
    'macro_f1': float(macro_f1),
    'weighted_precision': float(precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[0]),
    'weighted_recall': float(precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[1]),
    'weighted_f1': float(precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[2]),
}

# Per-class metrics
per_class = {}
for i in range(len(precision)):
    per_class[f'Class_{i}'] = {
        'precision': float(precision[i]),
        'recall': float(recall[i]),
        'f1': float(f1[i]),
        'support': int(support[i])
    }
metrics['per_class'] = per_class
metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

with open(output_dir / 'test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to {output_dir / 'test_metrics.json'}")
print("\nDone!")
