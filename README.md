# Network Intrusion Detection: Tree Ensembles vs Deep Learning

**Author:** Hamza Arshad  
**Date:** August 2026

---

## 📋 Project Overview

This project presents a comprehensive comparison of Random Forest, XGBoost, and 1D Convolutional Neural Networks (CNN) for flow-based network intrusion detection on the CICIDS2017 dataset. The study evaluates both detection quality (macro-F1) and deployment feasibility (training time, model size, inference latency) under severe class imbalance.

**Key Finding:** XGBoost achieves 85.55% macro-F1, outperforming Random Forest (83.34%) and optimized CNN (61.13%) by 24 percentage points, demonstrating superior sample efficiency on imbalanced data.

---

## 📁 Project Structure

```
25207938_Mohammed_Abdul_Razzak_Baig/
│
├── IEEE_PAPER_FINAL.pdf           # Final report (10 pages, single-column)
├── 25207938_Mohammed_Abdul_Razzak_Baig_proposal.pdf  # Part 1 proposal
│
├── src/                           # Source code
│   ├── data/
│   │   ├── preprocessing.py       # Data cleaning & preprocessing
│   │   └── data_loader.py         # Load train/val/test splits
│   ├── models/
│   │   ├── random_forest.py       # Random Forest implementation
│   │   ├── xgboost_model.py       # XGBoost implementation
│   │   ├── cnn_1d.py             # 1D-CNN with Focal Loss
│   │   └── imbalance.py          # Class weighting utilities
│   ├── training/
│   │   ├── train_baselines.py    # Train RF & XGBoost
│   │   └── train_cnn.py          # Train CNN
│   └── evaluation/
│       └── metrics.py            # Evaluation metrics
│
├── data/                          # Dataset files
│   ├── raw/cicids2017/           # Original CICIDS2017 CSV files (8 files)
│   ├── processed/cicids2017_preprocessor/  # Preprocessing artifacts
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   ├── medians.pkl
│   │   └── metadata.json
│   └── splits/cicids2017/        # Train/val/test splits
│       ├── X_train.npy           # 1,563,774 samples × 77 features
│       ├── y_train.npy
│       ├── X_val.npy             # 335,094 samples
│       ├── y_val.npy
│       ├── X_test.npy            # 335,095 samples
│       └── y_test.npy
│
├── models/                        # Trained models
│   └── cicids2017/
│       ├── random_forest/
│       │   ├── model.pkl         # 294 MB
│       │   └── test_metrics.json
│       └── xgboost/
│           ├── model.pkl         # 50 MB
│           └── test_metrics.json
│
├── outputs/                       # Experimental results
│   ├── xgboost_results.json      # 85.55% F1 (BEST)
│   ├── random_forest_results.json # 83.34% F1
│   ├── cnn_improved_results.json  # 61.13% F1
│   ├── best_cnn_improved.pt       # CNN model weights (635 KB)
│   └── cnn_improved_confusion_matrix.npy
│
├── paper_figures/                 # Visualizations (300 DPI)
│   ├── figure1_model_comparison.png
│   ├── figure2_perclass_heatmap.png
│   ├── figure3_confusion_matrix.png
│   ├── figure4_f1_vs_size.png
│   └── figure5_precision_recall.png
│
├── colab_train_cnn.py            # Standalone Colab training script
└── create_paper_figures.py       # Generate all visualizations
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
xgboost==2.0.0
torch==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
```

### Installation

```bash
# Clone or extract the project
cd "25207938_Mohammed_Abdul_Razzak_Baig (11 august)"

# Install dependencies
pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn
```

---

## 📊 Dataset Information

### Source
**CICIDS2017** - Canadian Institute for Cybersecurity IDS Dataset  

**⚠️ IMPORTANT - Dataset Not Included:**  
Due to file size constraints (4.1 GB), the raw dataset is **NOT included** in this submission.

**To run the code, download CICIDS2017 from:**  
🔗 https://www.unb.ca/cic/datasets/ids-2017.html

**Setup Instructions:**
1. Download the "MachineLearningCSV.zip" file (2.8 GB)
2. Extract the 8 CSV files
3. Place them in: `data/raw/cicids2017/`
4. Run preprocessing: `python src/data/preprocessing.py`

**Note:** All preprocessed artifacts (`scaler.pkl`, `label_encoder.pkl`, etc.) are included in `data/processed/` for reference.

### Dataset Statistics (After Preprocessing)
- **Total flows:** 2,233,963 (from original 2,830,743)
- **Features:** 77 (after removing leak features and duplicates)
- **Classes:** 15 (14 attacks + benign)
- **Train/Val/Test split:** 70%/15%/15% (stratified)
- **Class distribution:** Severely imbalanced (84.94% benign)

### Attack Types Included
1. BENIGN (normal traffic)
2. Brute Force (FTP, SSH)
3. DoS Hulk
4. PortScan
5. DDoS
6. DoS GoldenEye
7. DoS Slowloris
8. DoS Slowhttptest
9. Heartbleed
10. Infiltration
11. Web Attack (Brute Force)
12. Web Attack (XSS)
13. Web Attack (SQL Injection)
14. Botnet
15. Web Attack (additional)

**Extreme Imbalance Challenge:**
- Smallest class: 8 training samples
- 5 classes with <100 samples
- Largest class (Benign): 1,327,589 samples

---

## 🔧 Running the Code

### Step 1: Data Preprocessing (Already Done)

The preprocessed data is included. If you need to reprocess:

```bash
python src/data/preprocessing.py
```

This will:
- Load raw CSV files from `data/raw/cicids2017/`
- Remove duplicates (21% of dataset)
- Handle infinite/missing values
- Remove leak features (IP, ports, timestamps)
- Apply StandardScaler normalization
- Save splits to `data/splits/cicids2017/`

### Step 2: Train Random Forest

```bash
python src/training/train_baselines.py --model rf
```

**Training time:** ~15 minutes  
**Output:** `models/cicids2017/random_forest/model.pkl` (294 MB)  
**Results:** `models/cicids2017/random_forest/test_metrics.json`

### Step 3: Train XGBoost

```bash
python src/training/train_baselines.py --model xgboost
```

**Training time:** ~8 minutes  
**Output:** `models/cicids2017/xgboost/model.pkl` (50 MB)  
**Results:** `models/cicids2017/xgboost/test_metrics.json`

### Step 4: Train CNN (Local - CPU)

```bash
python src/training/train_cnn.py
```

**Training time:** ~3 hours (CPU), ~12 minutes (GPU)  
**Output:** `models/cicids2017/cnn/model.pt`  
**Results:** `models/cicids2017/cnn/test_metrics.json`

**Note:** CNN training is very slow on CPU. Use Colab for GPU training.

### Step 5: Train CNN (Google Colab - GPU)

1. Upload `colab_train_cnn.py` to Google Colab
2. Upload data splits to Google Drive: `data/splits/cicids2017/`
3. Update data path in the script
4. Run all cells

**Training time:** ~12 minutes (Tesla T4 GPU)  
**Output:** Downloads `best_cnn_improved.pt` and `cnn_improved_results.json`

### Step 6: Generate Visualizations

```bash
python create_paper_figures.py
```

**Output:** 6 figures saved to `paper_figures/` at 300 DPI

---

## 📈 Results Summary

### Overall Performance (Test Set)

| Model | Accuracy | Macro-Precision | Macro-Recall | **Macro-F1** | Model Size |
|-------|----------|-----------------|--------------|--------------|------------|
| **XGBoost** | **99.88%** | **91.98%** | 83.46% | **85.55%** 🏆 | 50 MB |
| Random Forest | 99.55% | 84.88% | **87.81%** | 83.34% | 294 MB |
| CNN (Optimized) | 97.01% | 63.32% | 74.56% | 61.13% | 0.64 MB |

### Key Findings

1. **XGBoost achieves best overall performance** with 85.55% macro-F1
2. **Tree ensembles outperform deep learning by 24 percentage points**
3. **CNN requires 102% F1 improvement through optimization** (from 30.24% to 61.13%)
4. **Classes with <100 samples challenge all models**, but tree ensembles degrade more gracefully
5. **XGBoost offers best deployment tradeoff**: highest F1, fast training (8 min), fast CPU inference (30 ms)

### Per-Class Performance Highlights

| Attack Type | Support | RF F1 | XGBoost F1 | CNN F1 |
|-------------|---------|-------|------------|--------|
| BENIGN | 284,619 | 99.76% | **99.95%** | 98.24% |
| DoS Hulk | 19,203 | 99.88% | **99.98%** | 89.05% |
| DDoS | 25,928 | 99.53% | **99.83%** | 95.16% |
| Brute Force FTP | 216 | 31.66% | **77.75%** | 45.59% |
| SQL Injection | 220 | 68.17% | **75.00%** | 21.30% |
| Infiltration | 5 | 57.14% | 33.33% | **4.08%** |
| Botnet | 3 | 66.67% | **80.00%** | **0.00%** |

**Observation:** CNN achieves 0% F1 on classes with <10 samples, while tree ensembles maintain 33-80% F1.

---

## 💻 System Requirements

### Minimum (for RF & XGBoost)
- **CPU:** 4 cores
- **RAM:** 16 GB
- **Storage:** 10 GB
- **OS:** macOS, Linux, or Windows

### Recommended (for CNN training)
- **GPU:** NVIDIA GPU with 8GB+ VRAM (Tesla T4, RTX 3060, or better)
- **RAM:** 16 GB
- **Storage:** 10 GB
- **Platform:** Google Colab (free T4 GPU) or local machine with CUDA

### Tested Environment
- **Hardware:** Apple M1, NVIDIA Tesla T4 (Colab)
- **OS:** macOS, Ubuntu 20.04
- **Python:** 3.11
- **CUDA:** 11.8 (for GPU training)

---

## 🔍 Code Structure Details

### Data Processing (`src/data/`)

**preprocessing.py**
- Loads 8 CSV files from CICIDS2017
- Removes 596,780 duplicate records (21%)
- Handles infinite values (replace with column max)
- Median imputation for missing values
- Removes leak features: IP addresses, ports, timestamps, flow IDs
- StandardScaler normalization (fit on train only)
- Stratified train/val/test split (70/15/15)
- Saves splits as `.npy` files for fast loading

**data_loader.py**
- Fast loading of preprocessed `.npy` splits
- Memory-efficient batch loading for large datasets
- Returns NumPy arrays or PyTorch tensors

### Models (`src/models/`)

**random_forest.py**
- RandomForestClassifier with 500 trees
- `class_weight='balanced'` for imbalance handling
- `bootstrap=True` for ensemble diversity
- `max_features='sqrt'` for decorrelation
- Parallel training with `n_jobs=-1`

**xgboost_model.py**
- XGBClassifier with 300 estimators
- `max_depth=5` (prevents overfitting on rare classes)
- `tree_method='hist'` (fast histogram-based training)
- `learning_rate=0.1` with early stopping
- Balanced class weights via `scale_pos_weight`

**cnn_1d.py**
- Custom 1D-CNN architecture:
  - Input: (batch, 1, 77 features)
  - Conv Block 1: 64 filters, kernel=3, MaxPool
  - Conv Block 2: 128 filters, kernel=3, MaxPool
  - Conv Block 3: 256 filters, kernel=3, AdaptiveAvgPool
  - FC: 256 → 128 → 15 classes
  - Dropout: 0.3 (regularization)
  - BatchNorm after each conv layer
- Focal Loss with γ=1.5 for imbalance
- sqrt(balanced) class weights
- Total parameters: 159,247

**imbalance.py**
- Focal Loss implementation
- Class weight computation (balanced, sqrt-balanced)
- SMOTE oversampling (optional, not used in final)

### Training (`src/training/`)

**train_baselines.py**
- Trains Random Forest and XGBoost
- Hyperparameter configurations
- Saves trained models and metrics
- Supports command-line arguments

**train_cnn.py**
- Trains 1D-CNN with PyTorch
- Implements training loop with validation
- Early stopping (patience=10)
- Learning rate scheduling (ReduceLROnPlateau)
- Saves best model by validation F1
- Full training history tracking

### Evaluation (`src/evaluation/`)

**metrics.py**
- Computes macro/weighted metrics
- Per-class precision, recall, F1
- Confusion matrix generation
- ROC-AUC scores
- Latency benchmarking

---

## 🎨 Visualizations

All figures are generated at 300 DPI for publication quality.

### Figure 1: Model Comparison
Bar chart comparing accuracy, precision, recall, and F1 across all three models.

### Figure 2: Per-Class Heatmap
Heatmap showing F1 scores for all 15 classes across RF, XGBoost, and CNN. Darker = better.

### Figure 3: Confusion Matrices
Side-by-side confusion matrices for all three models showing prediction patterns.

### Figure 4: F1 vs Sample Size
Scatter plot revealing the 100-sample threshold below which all models struggle.

### Figure 5: Precision-Recall Tradeoff
Scatter plot comparing precision vs recall for each model across all classes.

---

## 📝 Files Overview

### Report
- **IEEE_PAPER_FINAL.pdf** - Final report (10 pages, single-column IEEE format)
  - Includes all methodology, results, discussion, and references
  - 5 figures integrated
  - 3 tables with results
  - 12 references

### Code Files
- **colab_train_cnn.py** - Standalone script for Google Colab with GPU
- **create_paper_figures.py** - Generates all 6 visualizations from results

### Results Files (in `outputs/`)
- **xgboost_results.json** - Complete XGBoost metrics (85.55% F1)
- **random_forest_results.json** - Complete RF metrics (83.34% F1)
- **cnn_improved_results.json** - Complete CNN metrics (61.13% F1)
- **best_cnn_improved.pt** - Trained CNN model weights (PyTorch)
- **cnn_improved_confusion_matrix.npy** - 15×15 confusion matrix

---

## 🎓 Submission Components

This project includes:

1. ✅ **Final Report** - IEEE_PAPER_FINAL.pdf (10 pages, double-column)
2. ✅ **Source Code** - Complete implementation in `src/`
3. ✅ **Trained Models** - All 3 models with metrics
4. ✅ **Preprocessed Artifacts** - Scaler, encoder, metadata in `data/processed/`
5. ✅ **Results** - JSON files with complete metrics in `outputs/`
6. ✅ **Visualizations** - 6 publication-quality figures (300 DPI)
7. ✅ **Documentation** - This README

**⚠️ NOT INCLUDED (Due to Size):**
- ❌ **Raw Data** - `data/raw/` (2.8 GB) - Download from link above
- ❌ **Data Splits** - `data/splits/` (1.3 GB) - Regenerate with preprocessing script

**Data is publicly available** - Download CICIDS2017 from:  
🔗 https://www.unb.ca/cic/datasets/ids-2017.html

---

## 🔬 Reproducibility

All experiments use fixed random seeds (42) for reproducibility.

To reproduce results:

```bash
# 1. Data already preprocessed (deterministic)
# 2. Train Random Forest
python src/training/train_baselines.py --model rf --seed 42

# 3. Train XGBoost
python src/training/train_baselines.py --model xgboost --seed 42

# 4. Train CNN (GPU recommended)
python src/training/train_cnn.py --seed 42

# 5. Generate figures
python create_paper_figures.py
```

**Expected results** (may vary slightly due to hardware):
- XGBoost: 85.5% ± 0.5% macro-F1
- Random Forest: 83.3% ± 0.5% macro-F1
- CNN: 61.1% ± 1.0% macro-F1

---

## 📚 References

Dataset: I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018.

See IEEE_PAPER_FINAL.pdf for complete references.

---

## 📧 Contact

**Hamza Arshad**

---

## 📄 License

For academic and educational purposes only.

---

**Last Updated:** August 10, 2026
