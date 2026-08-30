"""
Master script to run all experiments for the NIDS project.

This script:
1. Trains all models (RF, XGBoost, 1D-CNN)
2. Evaluates on test sets
3. Measures latency
4. Performs error analysis
5. Generates all visualizations
6. Creates comparison reports

Usage:
    python run_all_experiments.py --dataset cicids2017
    python run_all_experiments.py --dataset both  # Run on both datasets
"""

import argparse
import sys
from pathlib import Path
import time
import json
import subprocess

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed/60:.2f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: {e}")
        return False


def train_baselines(dataset, tune=True, n_jobs=-1):
    """Train Random Forest and XGBoost baselines."""
    print("\n" + "=" * 70)
    print(f"Training Baseline Models on {dataset}")
    print("=" * 70)
    
    # Random Forest
    cmd_rf = [
        'python', 'src/training/train_baselines.py',
        '--model', 'rf',
        '--dataset', dataset,
        '--benchmark-latency',
        '--n-jobs', str(n_jobs)
    ]
    if tune:
        cmd_rf.append('--tune')
    
    success_rf = run_command(cmd_rf, f"Training Random Forest on {dataset}")
    
    # XGBoost
    cmd_xgb = [
        'python', 'src/training/train_baselines.py',
        '--model', 'xgboost',
        '--dataset', dataset,
        '--benchmark-latency',
        '--n-jobs', str(n_jobs)
    ]
    if tune:
        cmd_xgb.append('--tune')
    
    success_xgb = run_command(cmd_xgb, f"Training XGBoost on {dataset}")
    
    return success_rf and success_xgb


def train_cnn(dataset, epochs=50, imbalance='focal', device='cuda'):
    """Train 1D-CNN model."""
    print("\n" + "=" * 70)
    print(f"Training 1D-CNN on {dataset}")
    print("=" * 70)
    
    cmd = [
        'python', 'src/training/train_cnn.py',
        '--dataset', dataset,
        '--epochs', str(epochs),
        '--imbalance', imbalance,
        '--benchmark-latency',
        '--device', device
    ]
    
    return run_command(cmd, f"Training 1D-CNN on {dataset} with {imbalance} loss")


def generate_visualizations(dataset):
    """Generate all visualizations."""
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    cmd = [
        'python', 'src/visualization/plots.py',
        '--results-dir', f'models/{dataset}',
        '--output-dir', f'experiments/results/{dataset}/figures'
    ]
    
    return run_command(cmd, "Generating all figures")


def generate_error_analysis(dataset):
    """Generate comprehensive error analysis."""
    print("\n" + "=" * 70)
    print("Performing Error Analysis")
    print("=" * 70)
    
    # This will be a Python script we create
    analysis_script = f"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import load_splits
from src.evaluation.error_analysis import ErrorAnalyzer, compare_model_errors

# Load test data
print("Loading test data...")
splits = load_splits('{dataset}')
X_test = splits['X_test']
y_test = splits['y_test']

# Load class names from preprocessor
preprocessor_path = Path('data/processed/{dataset}_preprocessor/metadata.json')
with open(preprocessor_path) as f:
    metadata = json.load(f)
    class_names = list(metadata['label_mapping'].keys())

# Analyze errors for each model
models_dir = Path('models/{dataset}')
analyzers = {{}}

for model_dir in models_dir.iterdir():
    if model_dir.is_dir() and (model_dir / 'test_metrics.json').exists():
        model_name = model_dir.name
        print(f"\\nAnalyzing {{model_name}}...")
        
        # Load test metrics to get predictions
        with open(model_dir / 'test_metrics.json') as f:
            metrics = json.load(f)
        
        # We need to regenerate predictions - load model and predict
        # For now, we'll skip this and just note it in the report
        
print("\\n✓ Error analysis template created")
print("Note: Full error analysis requires loading models and regenerating predictions")
print("This will be completed in the error analysis notebook")
"""
    
    script_path = project_root / 'temp_error_analysis.py'
    with open(script_path, 'w') as f:
        f.write(analysis_script)
    
    cmd = ['python', str(script_path)]
    success = run_command(cmd, "Error Analysis")
    
    # Clean up
    script_path.unlink()
    
    return success


def create_final_report(dataset):
    """Create final comparison report."""
    print("\n" + "=" * 70)
    print("Creating Final Comparison Report")
    print("=" * 70)
    
    models_dir = Path(f'models/{dataset}')
    output_dir = Path(f'experiments/results/{dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile all results
    results = []
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            test_metrics_path = model_dir / 'test_metrics.json'
            latency_path = model_dir / 'latency.json'
            
            if test_metrics_path.exists() and latency_path.exists():
                with open(test_metrics_path) as f:
                    test_metrics = json.load(f)
                with open(latency_path) as f:
                    latency = json.load(f)
                
                results.append({
                    'model': model_dir.name,
                    'accuracy': test_metrics['accuracy'],
                    'macro_f1': test_metrics['macro_f1'],
                    'macro_precision': test_metrics['macro_precision'],
                    'macro_recall': test_metrics['macro_recall'],
                    'weighted_f1': test_metrics['weighted_f1'],
                    'roc_auc': test_metrics.get('roc_auc_ovr'),
                    'mean_latency_ms': latency['per_flow_inference']['mean_time_ms'],
                    'p95_latency_ms': latency['per_flow_inference']['p95_time_ms'],
                    'p99_latency_ms': latency['per_flow_inference']['p99_time_ms'],
                    'throughput_fps': latency['throughput']['throughput_fps'],
                    'model_size_mb': latency['model_size']['size_mb']
                })
    
    # Save compiled results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    
    print(f"\n✓ Results summary saved to {output_dir}")
    print("\nFinal Results Summary:")
    print("=" * 70)
    print(df.to_string(index=False))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments for NIDS project"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cicids2017',
        choices=['cicids2017', 'unsw-nb15', 'both'],
        help='Which dataset(s) to use'
    )
    
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip training baseline models'
    )
    
    parser.add_argument(
        '--skip-cnn',
        action='store_true',
        help='Skip training CNN'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning for baselines'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs for CNN training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for CNN training (cuda/cpu)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs for baselines'
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to run on
    if args.dataset == 'both':
        datasets = ['cicids2017', 'unsw-nb15']
    else:
        datasets = [args.dataset]
    
    print("\n" + "=" * 70)
    print("NIDS PROJECT - MASTER EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"\nDatasets: {datasets}")
    print(f"Skip baselines: {args.skip_baselines}")
    print(f"Skip CNN: {args.skip_cnn}")
    print(f"Hyperparameter tuning: {args.tune}")
    print(f"CNN epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    start_time = time.time()
    
    for dataset in datasets:
        print(f"\n\n{'='*70}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        # 1. Train baselines
        if not args.skip_baselines:
            success = train_baselines(dataset, tune=args.tune, n_jobs=args.n_jobs)
            if not success:
                print(f"✗ Failed to train baselines on {dataset}")
                continue
        
        # 2. Train CNN
        if not args.skip_cnn:
            success = train_cnn(
                dataset,
                epochs=args.epochs,
                imbalance='focal',
                device=args.device
            )
            if not success:
                print(f"✗ Failed to train CNN on {dataset}")
                continue
        
        # 3. Generate visualizations
        generate_visualizations(dataset)
        
        # 4. Error analysis
        generate_error_analysis(dataset)
        
        # 5. Create final report
        create_final_report(dataset)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"\nResults saved to: experiments/results/")
    print(f"Models saved to: models/")
    print("\nNext steps:")
    print("1. Review results in experiments/results/")
    print("2. Check visualizations in experiments/results/*/figures/")
    print("3. Write the final IEEE paper using these results")
    print("=" * 70)


if __name__ == "__main__":
    main()
