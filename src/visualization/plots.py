import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


def plot_quality_latency_tradeoff(results_df, output_path=None, title='Quality-Latency Tradeoff'):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color map for different model types
    colors = {
        'Random Forest': '#2ecc71',
        'XGBoost': '#e74c3c',
        '1D-CNN': '#3498db',
        'RF': '#2ecc71',
        'XGBoost': '#e74c3c',
        'CNN': '#3498db'
    }
    
    for idx, row in results_df.iterrows():
        model_name = row['Model']
        color = colors.get(model_name, '#95a5a6')
        
        ax.scatter(
            row['Latency (ms)'],
            row['Macro-F1'],
            s=200,
            c=color,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            label=model_name
        )
        
        # Add annotations
        ax.annotate(
            model_name,
            (row['Latency (ms)'], row['Macro-F1']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
        )
    
    ax.set_xlabel('Mean Inference Time per Flow (ms)', fontsize=12)
    ax.set_ylabel('Macro-F1 Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add ideal region (high quality, low latency)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Target F1=0.8')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()
    return fig


def plot_confusion_matrix(cm, class_names, model_name, output_path=None, normalize=False):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'
    
    # Adjust figure size based on number of classes
    n_classes = len(class_names)
    figsize = (max(8, n_classes * 0.8), max(6, n_classes * 0.6))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax,
        square=True
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    
    plt.show()
    return fig


def plot_per_class_recall(results_dict, output_path=None, title='Per-Class Recall Comparison'):
    
    # Convert to DataFrame
    data = []
    for model_name, class_metrics in results_dict.items():
        for class_name, recall in class_metrics.items():
            data.append({
                'Model': model_name,
                'Class': class_name,
                'Recall': recall
            })
    
    df = pd.DataFrame(data)
    
    # Get unique classes and models
    classes = df['Class'].unique()
    n_classes = len(classes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.8), 6))
    
    # Create grouped bar chart
    x = np.arange(n_classes)
    width = 0.25
    models = df['Model'].unique()
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        recalls = [model_data[model_data['Class'] == cls]['Recall'].values[0] 
                   if len(model_data[model_data['Class'] == cls]) > 0 else 0
                   for cls in classes]
        
        ax.bar(x + i * width, recalls, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Attack Class', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class recall plot to {output_path}")
    
    plt.show()
    return fig


def plot_class_distribution(y, class_names=None, output_path=None, title='Class Distribution'):
    
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in unique]
    
    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_classes)))
    bars = ax.bar(range(len(sorted_classes)), sorted_counts, color=colors, alpha=0.8)
    
    ax.set_xlabel('Attack Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels(sorted_classes, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    total = sum(sorted_counts)
    for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
        percentage = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sorted_counts) * 0.01,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution plot to {output_path}")
    
    plt.show()
    return fig


def plot_training_history(history, output_path=None):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val Macro-F1', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro-F1')
    axes[2].set_title('Validation Macro-F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {output_path}")
    
    plt.show()
    return fig


def plot_model_comparison_table(results_df, output_path=None):
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Format numbers
    formatted_df = results_df.copy()
    for col in formatted_df.columns:
        if col != 'Model' and formatted_df[col].dtype in [float, np.float64]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(formatted_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(formatted_df) + 1):
        for j in range(len(formatted_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison table to {output_path}")
    
    plt.show()
    return fig


def generate_all_figures(results_dir, output_dir):
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating All Figures for Paper")
    print("=" * 70)
    
    # Load results from all models
    models = {}
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            
            # Load metrics
            test_metrics_path = model_dir / 'test_metrics.json'
            latency_path = model_dir / 'latency.json'
            
            if test_metrics_path.exists() and latency_path.exists():
                with open(test_metrics_path) as f:
                    test_metrics = json.load(f)
                with open(latency_path) as f:
                    latency = json.load(f)
                
                models[model_name] = {
                    'metrics': test_metrics,
                    'latency': latency
                }
    
    if not models:
        print("No model results found. Please run experiments first.")
        return
    
    print(f"\nFound results for {len(models)} models: {list(models.keys())}")
    
    # 1. Quality-Latency Tradeoff
    print("\n1. Generating Quality-Latency Tradeoff Plot...")
    comparison_data = []
    for model_name, data in models.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Macro-F1': data['metrics']['macro_f1'],
            'Latency (ms)': data['latency']['per_flow_inference']['mean_time_ms']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    plot_quality_latency_tradeoff(
        df_comparison,
        output_path=output_dir / 'quality_latency_tradeoff.png'
    )
    
    # 2. Confusion Matrices
    print("\n2. Generating Confusion Matrices...")
    for model_name, data in models.items():
        cm = np.array(data['metrics']['confusion_matrix'])
        class_names = list(data['metrics']['per_class'].keys())
        
        plot_confusion_matrix(
            cm,
            class_names,
            model_name.replace('_', ' ').title(),
            output_path=output_dir / f'confusion_matrix_{model_name}.png',
            normalize=True
        )
    
    # 3. Per-Class Recall Comparison
    print("\n3. Generating Per-Class Recall Comparison...")
    recall_data = {}
    for model_name, data in models.items():
        recall_data[model_name.replace('_', ' ').title()] = {
            cls: metrics['recall']
            for cls, metrics in data['metrics']['per_class'].items()
        }
    
    plot_per_class_recall(
        recall_data,
        output_path=output_dir / 'per_class_recall_comparison.png'
    )
    
    # 4. Model Comparison Table
    print("\n4. Generating Model Comparison Table...")
    table_data = []
    for model_name, data in models.items():
        table_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Macro-F1': data['metrics']['macro_f1'],
            'Accuracy': data['metrics']['accuracy'],
            'Latency (ms)': data['latency']['per_flow_inference']['mean_time_ms'],
            'P95 (ms)': data['latency']['per_flow_inference']['p95_time_ms'],
            'Throughput (fps)': data['latency']['throughput']['throughput_fps'],
            'Size (MB)': data['latency']['model_size']['size_mb']
        })
    
    df_table = pd.DataFrame(table_data)
    plot_model_comparison_table(
        df_table,
        output_path=output_dir / 'model_comparison_table.png'
    )
    
    # Save as CSV too
    df_table.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print(f"✓ All figures generated and saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate all figures for paper")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='models/cicids2017',
        help='Directory containing model results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results/figures',
        help='Directory to save figures'
    )
    
    args = parser.parse_args()
    
    generate_all_figures(args.results_dir, args.output_dir)
