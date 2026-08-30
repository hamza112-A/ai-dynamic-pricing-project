import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class ErrorAnalyzer:
    
    
    def __init__(self, y_true, y_pred, y_pred_proba=None, class_names=None):
        
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = y_pred_proba
        
        self.n_classes = len(np.unique(y_true))
        
        if class_names is None:
            self.class_names = [f"Class_{i}" for i in range(self.n_classes)]
        else:
            self.class_names = class_names
        
        # Identify correct and incorrect predictions
        self.correct_mask = self.y_true == self.y_pred
        self.incorrect_mask = ~self.correct_mask
        
        self.correct_indices = np.where(self.correct_mask)[0]
        self.incorrect_indices = np.where(self.incorrect_mask)[0]
    
    def get_error_summary(self):
        
        total = len(self.y_true)
        correct = self.correct_mask.sum()
        incorrect = self.incorrect_mask.sum()
        
        summary = {
            'total_samples': total,
            'correct_predictions': int(correct),
            'incorrect_predictions': int(incorrect),
            'overall_accuracy': float(correct / total),
            'error_rate': float(incorrect / total)
        }
        
        return summary
    
    def analyze_per_class_errors(self):
        
        results = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get samples of this class
            class_mask = self.y_true == class_idx
            class_samples = class_mask.sum()
            
            if class_samples == 0:
                continue
            
            # Correct predictions for this class
            correct_for_class = (self.y_true == class_idx) & (self.y_pred == class_idx)
            correct_count = correct_for_class.sum()
            
            # Incorrect predictions for this class
            incorrect_for_class = (self.y_true == class_idx) & (self.y_pred != class_idx)
            incorrect_count = incorrect_for_class.sum()
            
            # Where were incorrect samples classified?
            if incorrect_count > 0:
                misclassified_as = self.y_pred[incorrect_for_class]
                misclass_distribution = {}
                for pred_class in np.unique(misclassified_as):
                    count = (misclassified_as == pred_class).sum()
                    misclass_distribution[self.class_names[pred_class]] = int(count)
            else:
                misclass_distribution = {}
            
            results[class_name] = {
                'total_samples': int(class_samples),
                'correct_predictions': int(correct_count),
                'incorrect_predictions': int(incorrect_count),
                'recall': float(correct_count / class_samples),
                'error_rate': float(incorrect_count / class_samples),
                'misclassified_as': misclass_distribution
            }
        
        return results
    
    def identify_most_confused_pairs(self, top_k=5):
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Get off-diagonal elements (actual confusions)
        confusions = []
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage_of_true': float(cm[i, j] / cm[i, :].sum() * 100)
                    })
        
        # Sort by count
        confusions = sorted(confusions, key=lambda x: x['count'], reverse=True)
        
        return confusions[:top_k]
    
    def analyze_confidence_distribution(self):
        
        if self.y_pred_proba is None:
            return None
        
        # Get max probability (confidence) for each prediction
        confidences = np.max(self.y_pred_proba, axis=1)
        
        correct_confidences = confidences[self.correct_mask]
        incorrect_confidences = confidences[self.incorrect_mask]
        
        return {
            'correct': {
                'mean': float(correct_confidences.mean()),
                'std': float(correct_confidences.std()),
                'median': float(np.median(correct_confidences)),
                'min': float(correct_confidences.min()),
                'max': float(correct_confidences.max())
            },
            'incorrect': {
                'mean': float(incorrect_confidences.mean()),
                'std': float(incorrect_confidences.std()),
                'median': float(np.median(incorrect_confidences)),
                'min': float(incorrect_confidences.min()),
                'max': float(incorrect_confidences.max())
            }
        }
    
    def analyze_rare_class_performance(self, threshold=0.05):
        
        total_samples = len(self.y_true)
        rare_classes = []
        
        per_class_errors = self.analyze_per_class_errors()
        
        for class_name, metrics in per_class_errors.items():
            class_proportion = metrics['total_samples'] / total_samples
            
            if class_proportion < threshold:
                rare_classes.append({
                    'class': class_name,
                    'samples': metrics['total_samples'],
                    'proportion': float(class_proportion),
                    'recall': metrics['recall'],
                    'error_rate': metrics['error_rate']
                })
        
        # Sort by recall (ascending) to find worst-performing rare classes
        rare_classes = sorted(rare_classes, key=lambda x: x['recall'])
        
        return rare_classes
    
    def generate_error_report(self, output_path=None):
        
        report = {}
        
        print("\n" + "=" * 70)
        print("Comprehensive Error Analysis Report")
        print("=" * 70)
        
        # 1. Overall error summary
        print("\n1. Overall Error Summary")
        print("-" * 70)
        summary = self.get_error_summary()
        report['overall_summary'] = summary
        
        print(f"Total samples: {summary['total_samples']:,}")
        print(f"Correct predictions: {summary['correct_predictions']:,} ({summary['overall_accuracy']*100:.2f}%)")
        print(f"Incorrect predictions: {summary['incorrect_predictions']:,} ({summary['error_rate']*100:.2f}%)")
        
        # 2. Per-class error analysis
        print("\n2. Per-Class Error Analysis")
        print("-" * 70)
        per_class = self.analyze_per_class_errors()
        report['per_class_errors'] = per_class
        
        print(f"{'Class':<25} {'Samples':>10} {'Recall':>10} {'Error Rate':>12}")
        print("-" * 70)
        for class_name, metrics in sorted(per_class.items(), key=lambda x: x[1]['recall']):
            print(f"{class_name:<25} {metrics['total_samples']:>10} "
                  f"{metrics['recall']:>10.4f} {metrics['error_rate']:>12.4f}")
        
        # 3. Most confused class pairs
        print("\n3. Most Confused Class Pairs")
        print("-" * 70)
        confused_pairs = self.identify_most_confused_pairs(top_k=10)
        report['confused_pairs'] = confused_pairs
        
        print(f"{'True Class':<25} {'Predicted As':<25} {'Count':>10} {'% of True':>12}")
        print("-" * 70)
        for pair in confused_pairs:
            print(f"{pair['true_class']:<25} {pair['predicted_class']:<25} "
                  f"{pair['count']:>10} {pair['percentage_of_true']:>12.2f}%")
        
        # 4. Confidence analysis
        if self.y_pred_proba is not None:
            print("\n4. Confidence Analysis")
            print("-" * 70)
            confidence_dist = self.analyze_confidence_distribution()
            report['confidence_distribution'] = confidence_dist
            
            print(f"\nCorrect predictions:")
            print(f"  Mean confidence: {confidence_dist['correct']['mean']:.4f}")
            print(f"  Median confidence: {confidence_dist['correct']['median']:.4f}")
            
            print(f"\nIncorrect predictions:")
            print(f"  Mean confidence: {confidence_dist['incorrect']['mean']:.4f}")
            print(f"  Median confidence: {confidence_dist['incorrect']['median']:.4f}")
            
            print(f"\nObservation: ", end="")
            if confidence_dist['incorrect']['mean'] > 0.5:
                print("Model is overconfident on errors!")
            else:
                print("Model has lower confidence on errors (good calibration)")
        
        # 5. Rare class performance
        print("\n5. Rare Class Performance (< 5% of dataset)")
        print("-" * 70)
        rare_classes = self.analyze_rare_class_performance(threshold=0.05)
        report['rare_class_performance'] = rare_classes
        
        if rare_classes:
            print(f"{'Class':<25} {'Samples':>10} {'Proportion':>12} {'Recall':>10}")
            print("-" * 70)
            for rc in rare_classes:
                print(f"{rc['class']:<25} {rc['samples']:>10} "
                      f"{rc['proportion']*100:>11.2f}% {rc['recall']:>10.4f}")
        else:
            print("No rare classes found.")
        
        print("\n" + "=" * 70)
        
        # Save report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✓ Report saved to {output_path}")
        
        return report


def compare_model_errors(analyzers_dict, output_path=None):
    
    print("\n" + "=" * 70)
    print("Cross-Model Error Comparison")
    print("=" * 70)
    
    comparison = {}
    
    # Get all class names
    first_analyzer = list(analyzers_dict.values())[0]
    class_names = first_analyzer.class_names
    
    # Compare per-class recall
    print("\n1. Per-Class Recall Comparison")
    print("-" * 70)
    
    recall_data = []
    for model_name, analyzer in analyzers_dict.items():
        per_class = analyzer.analyze_per_class_errors()
        for class_name in class_names:
            if class_name in per_class:
                recall_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Recall': per_class[class_name]['recall'],
                    'Samples': per_class[class_name]['total_samples']
                })
    
    df_recall = pd.DataFrame(recall_data)
    
    # Find classes where models differ most
    print("\nClasses with largest recall differences:")
    print("-" * 70)
    
    for class_name in class_names:
        class_data = df_recall[df_recall['Class'] == class_name]
        if len(class_data) > 0:
            max_recall = class_data['Recall'].max()
            min_recall = class_data['Recall'].min()
            diff = max_recall - min_recall
            
            if diff > 0.1:  # Significant difference
                best_model = class_data.loc[class_data['Recall'].idxmax(), 'Model']
                worst_model = class_data.loc[class_data['Recall'].idxmin(), 'Model']
                
                print(f"{class_name:<25} | Diff: {diff:.4f} | "
                      f"Best: {best_model} ({max_recall:.4f}) | "
                      f"Worst: {worst_model} ({min_recall:.4f})")
    
    # Compare on rare classes
    print("\n2. Rare Class Performance Comparison")
    print("-" * 70)
    
    for model_name, analyzer in analyzers_dict.items():
        rare_classes = analyzer.analyze_rare_class_performance(threshold=0.05)
        if rare_classes:
            avg_rare_recall = np.mean([rc['recall'] for rc in rare_classes])
            print(f"{model_name:<30} | Avg Rare Class Recall: {avg_rare_recall:.4f}")
    
    comparison['recall_comparison'] = df_recall.to_dict('records')
    
    # Save comparison
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        df_recall.to_csv(output_path.with_suffix('.csv'), index=False)
        
        print(f"\n✓ Comparison saved to {output_path}")
    
    return comparison


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing Error Analyzer...")
    
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Generate imbalanced data
    y_true = np.concatenate([
        np.zeros(500),  # 50% class 0
        np.ones(300),   # 30% class 1
        np.full(150, 2),  # 15% class 2
        np.full(40, 3),   # 4% class 3 (rare)
        np.full(10, 4)    # 1% class 4 (very rare)
    ]).astype(int)
    
    # Generate predictions with intentional errors
    y_pred = y_true.copy()
    # Add errors (20% error rate)
    error_indices = np.random.choice(len(y_true), size=200, replace=False)
    for idx in error_indices:
        y_pred[idx] = np.random.choice([i for i in range(n_classes) if i != y_true[idx]])
    
    # Generate probabilities
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    class_names = ['Benign', 'DoS', 'DDoS', 'Infiltration', 'Heartbleed']
    
    # Create analyzer
    analyzer = ErrorAnalyzer(y_true, y_pred, y_pred_proba, class_names)
    
    # Generate report
    report = analyzer.generate_error_report()
    
    print("\n✓ Error analysis test passed")
