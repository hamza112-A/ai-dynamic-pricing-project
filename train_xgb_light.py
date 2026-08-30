"""Quick XGBoost training with reduced parameters to avoid memory issues."""
import numpy as np
import json
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

print("Loading data...")
X_train = np.load('data/splits/cicids2017/X_train.npy')
y_train = np.load('data/splits/cicids2017/y_train.npy')
X_val = np.load('data/splits/cicids2017/X_val.npy')
y_val = np.load('data/splits/cicids2017/y_val.npy')
X_test = np.load('data/splits/cicids2017/X_test.npy')
y_test = np.load('data/splits/cicids2017/y_test.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Use lighter parameters to avoid crash
print("\nTraining XGBoost with reduced parameters...")
model = xgb.XGBClassifier(
    n_estimators=300,  # Reduced from 600
    max_depth=5,       # Reduced from 6
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,          # Reduced from -1
    tree_method='hist', # Faster method
    verbosity=1
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

print("\nEvaluating on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
macro_precision = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[0]
macro_recall = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[1]
macro_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[2]

print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# Save model
output_dir = Path('models/cicids2017/xgboost')
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, output_dir / 'model.pkl')
print(f"\nModel saved to {output_dir}")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'macro_precision': float(macro_precision),
    'macro_recall': float(macro_recall),
    'macro_f1': float(macro_f1),
    'weighted_precision': float(precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[0]),
    'weighted_recall': float(precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[1]),
    'weighted_f1': float(precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[2]),
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
metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

# ROC-AUC
try:
    metrics['roc_auc_ovr'] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro'))
    metrics['roc_auc_ovo'] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro'))
except:
    pass

with open(output_dir / 'test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to {output_dir / 'test_metrics.json'}")
print("\nDone!")
