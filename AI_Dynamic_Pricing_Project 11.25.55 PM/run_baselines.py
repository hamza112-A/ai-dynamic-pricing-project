

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from evaluation import BaselineStrategy, PerformanceEvaluator

print("="*80)
print("BASELINE STRATEGIES EVALUATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = 'data/processed_data.csv'
SAMPLE_SIZE = 10000  # Fast evaluation

print("[CONFIG] Baseline Evaluation:")
print(f"  Data: {DATA_PATH}")
print(f"  Sample size: {SAMPLE_SIZE:,}")
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

start_time = time.time()

df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {len(df):,} rows")

# Sample for speed
if len(df) > SAMPLE_SIZE:
    np.random.seed(42)
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled to: {len(df):,} rows")

# Split train/test  
split_idx = int(len(df) * 0.7)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"Test set: {len(test_df):,} rows")

# Base price
base_price = float(df['adr'].mean())
print(f"Base price: ${base_price:.2f}")

load_time = time.time() - start_time
print(f"\n✓ Data loaded in {load_time:.1f}s")
print()

# ============================================================================
# STEP 2: Evaluate Baselines
# ============================================================================

print("="*80)
print("STEP 2: EVALUATING BASELINE STRATEGIES")
print("="*80)

evaluator = PerformanceEvaluator()

# Strategy 1: Fixed Pricing
print("\n1. Fixed Pricing:")
result = BaselineStrategy.fixed_pricing(test_df, base_price)
evaluator.add_result("Fixed Pricing", result)
print(f"   Revenue: ${result['total_revenue']:,.2f}")
print(f"   Bookings: {result['bookings_made']:,}")
print(f"   Avg Price: ${result['avg_price']:.2f}")

# Strategy 2: Rule-Based
print("\n2. Rule-Based Pricing:")
result = BaselineStrategy.rule_based_pricing(test_df, base_price)
evaluator.add_result("Rule-Based", result)
print(f"   Revenue: ${result['total_revenue']:,.2f}")
print(f"   Bookings: {result['bookings_made']:,}")
print(f"   Avg Price: ${result['avg_price']:.2f}")

# Strategy 3: Historical Average
print("\n3. Historical Average Pricing:")
result = BaselineStrategy.historical_avg_pricing(test_df, base_price)
evaluator.add_result("Historical Average", result)
print(f"   Revenue: ${result['total_revenue']:,.2f}")
print(f"   Bookings: {result['bookings_made']:,}")
print(f"   Avg Price: ${result['avg_price']:.2f}")

print()

# ============================================================================
# STEP 3: Comparison
# ============================================================================

print("="*80)
print("STEP 3: COMPARISON TABLE")
print("="*80)

comparison_df = evaluator.compare("Fixed Pricing")
print(comparison_df.to_string(index=False))
print()

# ============================================================================
# STEP 4: Save Results
# ============================================================================

print("="*80)
print("STEP 4: SAVING RESULTS")
print("="*80)

import os
os.makedirs('results', exist_ok=True)

# Save results JSON
evaluator.save_results_json('results/baseline_results.json')

# Save comparison table
comparison_df.to_csv('results/baseline_comparison.csv', index=False)
print("[SAVE] Comparison table → results/baseline_comparison.csv")

# Create visualizations
evaluator.plot_comparison('results/figures')

print()

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

print("="*80)
print("✅ BASELINE EVALUATION COMPLETE!")
print("="*80)
print()
print(f"Total Runtime: {total_time:.1f}s")
print()
print("Strategies evaluated:")
print("  ✓ Fixed Pricing")
print("  ✓ Rule-Based Pricing")
print("  ✓ Historical Average Pricing")
print()
print("Files saved:")
print("  - results/baseline_results.json")
print("  - results/baseline_comparison.csv")
print("  - results/figures/strategy_comparison.png")
print()
print("Next steps:")
print("  1. Run DQN training: python run_super_fast_training.py")
print("  2. Run GA training: python run_ga_fast.py")
print("  3. Compare all results")
print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
