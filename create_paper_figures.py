"""
Generate all figures for IEEE paper from model results.
Run: python create_paper_figures.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path('paper_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading model results...")

# Load results
with open('models/cicids2017/random_forest/test_metrics.json', 'r') as f:
    rf_results = json.load(f)

with open('models/cicids2017/xgboost/test_metrics.json', 'r') as f:
    xgb_results = json.load(f)

with open('outputs/cnn_improved_results.json', 'r') as f:
    cnn_results = json.load(f)

print("✓ Loaded all results")

# ============================================================================
# FIGURE 1: Model Comparison - Macro Metrics
# ============================================================================
print("\nCreating Figure 1: Model Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['XGBoost', 'Random Forest', 'CNN (Improved)']
accuracy = [xgb_results['accuracy']*100, rf_results['accuracy']*100, cnn_results['accuracy']*100]
precision = [xgb_results['macro_precision']*100, rf_results['macro_precision']*100, cnn_results['macro_precision']*100]
recall = [xgb_results['macro_recall']*100, rf_results['macro_recall']*100, cnn_results['macro_recall']*100]
f1 = [xgb_results['macro_f1']*100, rf_results['macro_f1']*100, cnn_results['macro_f1']*100]

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3498db')
bars2 = ax.bar(x - 0.5*width, precision, width, label='Macro Precision', color='#e74c3c')
bars3 = ax.bar(x + 0.5*width, recall, width, label='Macro Recall', color='#2ecc71')
bars4 = ax.bar(x + 1.5*width, f1, width, label='Macro F1', color='#f39c12')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison on CICIDS2017', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 105])

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure1_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Figure 1")

# ============================================================================
# FIGURE 2: Per-Class F1 Scores Heatmap
# ============================================================================
print("Creating Figure 2: Per-Class Performance...")

fig, ax = plt.subplots(figsize=(12, 8))

# Extract per-class F1 scores
classes = list(range(15))
rf_f1_scores = [rf_results['per_class'][f'Class_{i}']['f1'] * 100 for i in classes]
xgb_f1_scores = [xgb_results['per_class'][f'Class_{i}']['f1'] * 100 for i in classes]
cnn_f1_scores = [cnn_results['per_class'][f'Class_{i}']['f1'] * 100 for i in classes]

# Create heatmap data
data = np.array([xgb_f1_scores, rf_f1_scores, cnn_f1_scores])

# Plot heatmap
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks and labels
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels([f'C{i}' for i in classes])
ax.set_yticklabels(models)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('F1 Score (%)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(models)):
    for j in range(len(classes)):
        text = ax.text(j, i, f'{data[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Per-Class F1 Scores by Model', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure2_perclass_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Figure 2")

# ============================================================================
# FIGURE 3: Confusion Matrix for Best Model (XGBoost)
# ============================================================================
print("Creating Figure 3: Confusion Matrix...")

fig, ax = plt.subplots(figsize=(14, 12))

cm = np.array(xgb_results['confusion_matrix'])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot
im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax.set_xticks(np.arange(15))
ax.set_yticks(np.arange(15))
ax.set_xticklabels([f'C{i}' for i in range(15)])
ax.set_yticklabels([f'C{i}' for i in range(15)])

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations (only for significant values)
for i in range(15):
    for j in range(15):
        if cm_normalized[i, j] > 5:  # Only show if > 5%
            text = ax.text(j, i, f'{cm_normalized[i, j]:.0f}',
                          ha="center", va="center",
                          color="white" if cm_normalized[i, j] > 50 else "black",
                          fontsize=7)

ax.set_title('XGBoost Confusion Matrix (Normalized by Row)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
ax.set_ylabel('True Class', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Figure 3")

# ============================================================================
# FIGURE 4: F1 Score vs Class Size
# ============================================================================
print("Creating Figure 4: F1 vs Sample Size...")

fig, ax = plt.subplots(figsize=(10, 6))

# Get sample sizes from support
supports = [xgb_results['per_class'][f'Class_{i}']['support'] for i in range(15)]

# Plot for each model
ax.scatter(supports, xgb_f1_scores, s=100, alpha=0.7, label='XGBoost', marker='o')
ax.scatter(supports, rf_f1_scores, s=100, alpha=0.7, label='Random Forest', marker='s')
ax.scatter(supports, cnn_f1_scores, s=100, alpha=0.7, label='CNN (Improved)', marker='^')

# Add trend line
ax.plot(supports, xgb_f1_scores, alpha=0.3, linestyle='--', color='blue')

ax.set_xlabel('Training Samples (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax.set_title('F1 Score vs Training Sample Size', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Annotate critical classes
critical_classes = [8, 9, 13]  # Smallest classes
for cls in critical_classes:
    support = supports[cls]
    f1 = xgb_f1_scores[cls]
    ax.annotate(f'C{cls}\n({support})', 
                xy=(support, f1), 
                xytext=(support*2, f1-10),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=8, color='red')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure4_f1_vs_size.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Figure 4")

# ============================================================================
# FIGURE 5: Precision-Recall Trade-off
# ============================================================================
print("Creating Figure 5: Precision-Recall Trade-off...")

fig, ax = plt.subplots(figsize=(8, 8))

# Plot points
models_plot = ['XGBoost', 'Random Forest', 'CNN (Improved)']
precisions = [xgb_results['macro_precision']*100, rf_results['macro_precision']*100, cnn_results['macro_precision']*100]
recalls = [xgb_results['macro_recall']*100, rf_results['macro_recall']*100, cnn_results['macro_recall']*100]
f1_scores_pr = [xgb_results['macro_f1']*100, rf_results['macro_f1']*100, cnn_results['macro_f1']*100]
colors = ['#3498db', '#e74c3c', '#2ecc71']
markers = ['o', 's', '^']
sizes = [200, 200, 200]

for i, model in enumerate(models_plot):
    ax.scatter(recalls[i], precisions[i], s=sizes[i], alpha=0.7, 
              color=colors[i], marker=markers[i], label=model, edgecolors='black', linewidth=2)
    
    # Add labels
    ax.annotate(f'{model}\nF1={f1_scores_pr[i]:.1f}%', 
                xy=(recalls[i], precisions[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

# Diagonal line (F1 iso-curves)
for f1_line in [30, 50, 70, 85]:
    r = np.linspace(1, 100, 100)
    p = (f1_line * r) / (2 * r - f1_line)
    p = np.clip(p, 0, 100)
    ax.plot(r, p, '--', alpha=0.3, color='gray', linewidth=1)
    ax.text(95, (f1_line * 95) / (2 * 95 - f1_line), f'F1={f1_line}%', 
            fontsize=8, color='gray', alpha=0.6)

ax.set_xlabel('Macro Recall (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro Precision (%)', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([20, 100])
ax.set_ylim([20, 100])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'figure5_precision_recall.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Figure 5")

# ============================================================================
# TABLE 1: Summary Statistics
# ============================================================================
print("\nCreating Table 1: Summary Statistics...")

table_data = {
    'Model': ['XGBoost', 'Random Forest', 'CNN (Improved)'],
    'Accuracy (%)': [f"{xgb_results['accuracy']*100:.2f}", 
                     f"{rf_results['accuracy']*100:.2f}", 
                     f"{cnn_results['accuracy']*100:.2f}"],
    'Macro-P (%)': [f"{xgb_results['macro_precision']*100:.2f}", 
                    f"{rf_results['macro_precision']*100:.2f}", 
                    f"{cnn_results['macro_precision']*100:.2f}"],
    'Macro-R (%)': [f"{xgb_results['macro_recall']*100:.2f}", 
                    f"{rf_results['macro_recall']*100:.2f}", 
                    f"{cnn_results['macro_recall']*100:.2f}"],
    'Macro-F1 (%)': [f"{xgb_results['macro_f1']*100:.2f}", 
                     f"{rf_results['macro_f1']*100:.2f}", 
                     f"{cnn_results['macro_f1']*100:.2f}"],
}

# Create table figure
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=[[table_data['Model'][i],
                           table_data['Accuracy (%)'][i],
                           table_data['Macro-P (%)'][i],
                           table_data['Macro-R (%)'][i],
                           table_data['Macro-F1 (%)'][i]] for i in range(3)],
                colLabels=['Model', 'Accuracy', 'Macro-P', 'Macro-R', 'Macro-F1'],
                cellLoc='center',
                loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best values
for i in range(1, 4):
    for j in range(5):
        if i == 1:  # XGBoost row
            table[(i, j)].set_facecolor('#d5f4e6')

plt.title('Table 1: Overall Model Performance on Test Set', fontsize=14, fontweight='bold', pad=20)
plt.savefig(OUTPUT_DIR / 'table1_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Table 1")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*70}")
print("ALL FIGURES CREATED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
print("\nGenerated files:")
print("  1. figure1_model_comparison.png - Overall metrics bar chart")
print("  2. figure2_perclass_heatmap.png - Per-class F1 heatmap")
print("  3. figure3_confusion_matrix.png - XGBoost confusion matrix")
print("  4. figure4_f1_vs_size.png - F1 vs sample size scatter")
print("  5. figure5_precision_recall.png - P-R trade-off")
print("  6. table1_summary.png - Summary statistics table")
print(f"\n{'='*70}")
print("Ready to insert into IEEE paper!")
print(f"{'='*70}\n")
