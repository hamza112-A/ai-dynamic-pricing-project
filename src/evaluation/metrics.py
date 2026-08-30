import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from scipy import stats
import json
from pathlib import Path


def compute_all_metrics(y_true, y_pred, y_pred_proba=None, class_names=None):
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class and averaged metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged metrics (all classes weighted equally)
    metrics['macro_precision'] = precision.mean()
    metrics['macro_recall'] = recall.mean()
    metrics['macro_f1'] = f1.mean()
    
    # Weighted-averaged metrics (weighted by support)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_precision'] = p_weighted
    metrics['weighted_recall'] = r_weighted
    metrics['weighted_f1'] = f1_weighted
    
    # Per-class metrics
    n_classes = len(precision)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # AUC metrics (if probabilities provided)
    if y_pred_proba is not None:
        try:
            # Multi-class ROC AUC (one-vs-rest)
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='macro'
            )
            metrics['roc_auc_ovo'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovo', average='macro'
            )
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            metrics['roc_auc_ovr'] = None
            metrics['roc_auc_ovo'] = None
        
        try:
            # Average Precision (PR AUC)
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            ap_scores = []
            for i in range(n_classes):
                ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                ap_scores.append(ap)
            
            metrics['average_precision_macro'] = np.mean(ap_scores)
            metrics['per_class_ap'] = {
                class_names[i]: float(ap_scores[i]) 
                for i in range(n_classes)
            }
        except Exception as e:
            print(f"Warning: Could not compute AP: {e}")
            metrics['average_precision_macro'] = None
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    
    if metrics.get('roc_auc_ovr') is not None:
        print(f"  ROC AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
    if metrics.get('average_precision_macro') is not None:
        print(f"  Avg Precision:     {metrics['average_precision_macro']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<20} "
              f"{class_metrics['precision']:>10.4f} "
              f"{class_metrics['recall']:>10.4f} "
              f"{class_metrics['f1']:>10.4f} "
              f"{class_metrics['support']:>10}")


def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95, random_state=42):
    
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metric
        score = metric_fn(y_true_boot, y_pred_boot)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Compute confidence interval
    alpha = 1 - ci
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean = scores.mean()
    
    return {
        'mean': mean,
        'lower': lower,
        'upper': upper,
        'std': scores.std()
    }


def compare_models_statistical(
    y_true,
    y_pred_model1,
    y_pred_model2,
    model1_name="Model 1",
    model2_name="Model 2",
    metric='f1_macro',
    n_bootstrap=1000,
    random_state=42
):
    
    from sklearn.metrics import f1_score
    
    # Define metric function
    if metric == 'accuracy':
        metric_fn = accuracy_score
    elif metric == 'f1_macro':
        metric_fn = lambda y_t, y_p: f1_score(y_t, y_p, average='macro', zero_division=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    print(f"\nStatistical Comparison: {model1_name} vs {model2_name}")
    print("=" * 70)
    
    # Compute base scores
    score1 = metric_fn(y_true, y_pred_model1)
    score2 = metric_fn(y_true, y_pred_model2)
    
    print(f"\n{model1_name} {metric}: {score1:.4f}")
    print(f"{model2_name} {metric}: {score2:.4f}")
    print(f"Difference: {score2 - score1:.4f}")
    
    # Bootstrap confidence intervals
    print(f"\nBootstrap CI ({n_bootstrap} samples):")
    
    ci1 = bootstrap_confidence_interval(
        y_true, y_pred_model1, metric_fn,
        n_bootstrap=n_bootstrap, random_state=random_state
    )
    ci2 = bootstrap_confidence_interval(
        y_true, y_pred_model2, metric_fn,
        n_bootstrap=n_bootstrap, random_state=random_state + 1
    )
    
    print(f"{model1_name}: {ci1['mean']:.4f} [{ci1['lower']:.4f}, {ci1['upper']:.4f}]")
    print(f"{model2_name}: {ci2['mean']:.4f} [{ci2['lower']:.4f}, {ci2['upper']:.4f}]")
    
    # Compute p-value using bootstrap
    np.random.seed(random_state)
    n_samples = len(y_true)
    diff_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred1_boot = y_pred_model1[indices]
        y_pred2_boot = y_pred_model2[indices]
        
        score1_boot = metric_fn(y_true_boot, y_pred1_boot)
        score2_boot = metric_fn(y_true_boot, y_pred2_boot)
        
        diff_scores.append(score2_boot - score1_boot)
    
    diff_scores = np.array(diff_scores)
    
    # Two-tailed p-value
    p_value = 2 * min(
        (diff_scores >= 0).mean(),
        (diff_scores < 0).mean()
    )
    
    print(f"\nStatistical Significance:")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.001:
        print(f"  Result: {model2_name} is statistically better (p < 0.001)")
    elif p_value < 0.01:
        print(f"  Result: {model2_name} is statistically better (p < 0.01)")
    elif p_value < 0.05:
        print(f"  Result: {model2_name} is statistically better (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    return {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_score': score1,
        'model2_score': score2,
        'difference': score2 - score1,
        'model1_ci': ci1,
        'model2_ci': ci2,
        'p_value': p_value
    }


def save_metrics(metrics, output_path):
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    metrics_clean = convert_types(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    
    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    # Quick test
    print("Testing metrics module...")
    
    # Generate dummy predictions
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    class_names = [f"Attack_{i}" for i in range(n_classes)]
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_pred_proba, class_names)
    
    # Print metrics
    print_metrics(metrics)
    
    # Test statistical comparison
    y_pred_model2 = np.random.randint(0, n_classes, n_samples)
    
    comparison = compare_models_statistical(
        y_true, y_pred, y_pred_model2,
        model1_name="Model A",
        model2_name="Model B",
        n_bootstrap=100  # Reduced for testing
    )
    
    print("\n✓ Metrics module test passed")
