import argparse
import sys
from pathlib import Path
import time
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_splits, get_class_distribution, compute_class_weights
from src.models.random_forest import RandomForestModel, tune_random_forest
from src.models.xgboost_model import XGBoostModel, tune_xgboost
from src.models.imbalance import apply_smote
from src.evaluation.metrics import compute_all_metrics, print_metrics, save_metrics
from src.evaluation.latency import comprehensive_latency_benchmark


def train_random_forest(args, X_train, y_train, X_val, y_val, X_test, y_test):
    
    print("\n" + "=" * 70)
    print("Training Random Forest")
    print("=" * 70)
    
    if args.tune:
        # Hyperparameter tuning
        model, best_params = tune_random_forest(
            X_train, y_train, X_val, y_val,
            scoring='f1_macro',
            n_jobs=args.n_jobs
        )
        
        # Wrap in our model class
        rf_model = RandomForestModel(**best_params, n_jobs=args.n_jobs)
        rf_model.model = model
    else:
        # Use default/specified parameters
        rf_model = RandomForestModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight=args.class_weight,
            n_jobs=args.n_jobs,
            random_state=args.random_seed
        )
        
        # Apply SMOTE if specified
        if args.imbalance == 'smote':
            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
            rf_model.fit(X_train_resampled, y_train_resampled)
        else:
            rf_model.fit(X_train, y_train)
    
    # Save model
    model_dir = Path(args.output_dir) / args.dataset / 'random_forest'
    rf_model.save(model_dir)
    
    # Evaluate
    print("\nEvaluating Random Forest...")
    
    # Validation set
    y_val_pred = rf_model.predict(X_val)
    y_val_proba = rf_model.predict_proba(X_val)
    val_metrics = compute_all_metrics(y_val, y_val_pred, y_val_proba)
    print_metrics(val_metrics, "Random Forest - Validation Set")
    
    # Test set
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)
    test_metrics = compute_all_metrics(y_test, y_test_pred, y_test_proba)
    print_metrics(test_metrics, "Random Forest - Test Set")
    
    # Save metrics
    save_metrics(val_metrics, model_dir / 'val_metrics.json')
    save_metrics(test_metrics, model_dir / 'test_metrics.json')
    
    # Latency benchmark
    if args.benchmark_latency:
        latency_results = comprehensive_latency_benchmark(
            rf_model,
            X_test,
            model_path=model_dir,
            model_name=f"Random Forest ({args.dataset})",
            save_path=model_dir / 'latency.json'
        )
    
    return rf_model, test_metrics


def train_xgboost(args, X_train, y_train, X_val, y_val, X_test, y_test):
    
    print("\n" + "=" * 70)
    print("Training XGBoost")
    print("=" * 70)
    
    if args.tune:
        # Hyperparameter tuning
        model, best_params = tune_xgboost(
            X_train, y_train, X_val, y_val,
            scoring='f1_macro',
            n_jobs=args.n_jobs
        )
        
        # Wrap in our model class
        xgb_model = XGBoostModel(**best_params, n_jobs=args.n_jobs)
        xgb_model.model = model
    else:
        # Use default/specified parameters
        xgb_model = XGBoostModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_jobs=args.n_jobs,
            random_state=args.random_seed
        )
        
        # Apply SMOTE if specified
        if args.imbalance == 'smote':
            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
            xgb_model.fit(X_train_resampled, y_train_resampled, X_val, y_val)
        else:
            xgb_model.fit(X_train, y_train, X_val, y_val)
    
    # Save model
    model_dir = Path(args.output_dir) / args.dataset / 'xgboost'
    xgb_model.save(model_dir)
    
    # Evaluate
    print("\nEvaluating XGBoost...")
    
    # Validation set
    y_val_pred = xgb_model.predict(X_val)
    y_val_proba = xgb_model.predict_proba(X_val)
    val_metrics = compute_all_metrics(y_val, y_val_pred, y_val_proba)
    print_metrics(val_metrics, "XGBoost - Validation Set")
    
    # Test set
    y_test_pred = xgb_model.predict(X_test)
    y_test_proba = xgb_model.predict_proba(X_test)
    test_metrics = compute_all_metrics(y_test, y_test_pred, y_test_proba)
    print_metrics(test_metrics, "XGBoost - Test Set")
    
    # Save metrics
    save_metrics(val_metrics, model_dir / 'val_metrics.json')
    save_metrics(test_metrics, model_dir / 'test_metrics.json')
    
    # Latency benchmark
    if args.benchmark_latency:
        latency_results = comprehensive_latency_benchmark(
            xgb_model,
            X_test,
            model_path=model_dir,
            model_name=f"XGBoost ({args.dataset})",
            save_path=model_dir / 'latency.json'
        )
    
    return xgb_model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline models (RF, XGBoost)")
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['rf', 'xgboost', 'both'],
        help='Which model to train'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='cicids2017',
        choices=['cicids2017', 'unsw-nb15'],
        help='Which dataset to use'
    )
    
    # Hyperparameter tuning
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning with GridSearchCV'
    )
    
    # Imbalance handling
    parser.add_argument(
        '--imbalance',
        type=str,
        default='class_weight',
        choices=['none', 'class_weight', 'smote'],
        help='Imbalance handling strategy'
    )
    
    # Model hyperparameters
    parser.add_argument('--n-estimators', type=int, default=500, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=None, help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate (XGBoost)')
    parser.add_argument('--class-weight', type=str, default='balanced', help='Class weight (RF)')
    
    # Training settings
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')
    
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
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
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
    
    # Train models
    start_time = time.time()
    
    if args.model == 'rf' or args.model == 'both':
        train_random_forest(args, X_train, y_train, X_val, y_val, X_test, y_test)
    
    if args.model == 'xgboost' or args.model == 'both':
        train_xgboost(args, X_train, y_train, X_val, y_val, X_test, y_test)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete! Total time: {elapsed_time/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
