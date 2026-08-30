
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from genetic_algorithm import GeneticAlgorithmPricing

print("="*80)
print("FAST GENETIC ALGORITHM TRAINING")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = 'data/processed_data.csv'
POPULATION_SIZE = 50  # Smaller population
GENERATIONS = 30  # Fewer generations  
SAMPLE_SIZE = 5000  # Small dataset for speed

print("[CONFIG] Fast GA Configuration:")
print(f"  Population size: {POPULATION_SIZE}")
print(f"  Generations: {GENERATIONS}")
print(f"  Data sample size: {SAMPLE_SIZE:,}")
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

start_time = time.time()

# Load and sample data
df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {len(df):,} rows")

# Sample for speed
if len(df) > SAMPLE_SIZE:
    np.random.seed(42)
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled to: {len(df):,} rows")

# Split train/test
split_idx = int(len(df) * 0.7)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"Train set: {len(train_df):,} rows")
print(f"Test set: {len(test_df):,} rows")

load_time = time.time() - start_time
print(f"\n✓ Data loaded in {load_time:.1f}s")
print()

# ============================================================================
# STEP 2: Initialize GA
# ============================================================================

print("="*80)
print("STEP 2: INITIALIZING GENETIC ALGORITHM")
print("="*80)

base_price = float(train_df['adr'].mean())
print(f"Base price: ${base_price:.2f}")

ga = GeneticAlgorithmPricing(
    data=train_df,
    base_price=base_price,
    pop_size=POPULATION_SIZE,
    generations=GENERATIONS,
    crossover_prob=0.7,
    mutation_prob=0.2,
    mutation_sigma=0.1,
    tournament_k=3,
    n_scenarios=50,
    seed=42
)

print("✓ GA initialized")
print()

# ============================================================================
# STEP 3: Train (Evolve)
# ============================================================================

print("="*80)
print(f"STEP 3: EVOLUTION ({GENERATIONS} GENERATIONS)")
print("="*80)

train_start = time.time()

best_strategy, history = ga.evolve(verbose=True)

train_time = time.time() - train_start

print(f"\n✓ Evolution complete in {train_time/60:.1f} minutes")
print()

# ============================================================================
# STEP 4: Evaluate on Train Set
# ============================================================================

print("="*80)
print("STEP 4: TRAIN SET EVALUATION")
print("="*80)

train_results = ga.evaluate_strategy(best_strategy, verbose=True)
print()

# ============================================================================
# STEP 5: Evaluate on Test Set
# ============================================================================

print("="*80)
print("STEP 5: TEST SET EVALUATION")
print("="*80)

# Create new GA instance for test evaluation
ga_test = GeneticAlgorithmPricing(
    data=test_df,
    base_price=base_price,
    pop_size=10,  # Not used for evaluation
    generations=1,  # Not used for evaluation
    n_scenarios=50,
    seed=42
)

test_results = ga_test.evaluate_strategy(best_strategy, verbose=True)
print()

# ============================================================================
# STEP 6: Compare with Baselines
# ============================================================================

print("="*80)
print("STEP 6: BASELINE COMPARISON")
print("="*80)

# Scale results to 36 episodes for comparison
EVAL_EPISODES = 36
revenue_per_1k = test_results['total_revenue'] / len(test_df)
scaled_revenue = revenue_per_1k * 1000 * EVAL_EPISODES

print(f"GA Performance (scaled for comparison):")
print(f"  Test Revenue:       ${test_results['total_revenue']:>12,.2f}")
print(f"  Scaled Revenue:     ${scaled_revenue:>12,.2f}")
print(f"  Test Bookings:      {test_results['bookings_made']:>12,d}")
print(f"  Avg Price:          ${test_results['avg_price']:>12.2f}")
print(f"  Training Time:      {train_time/60:>12.1f} min")
print()

# Load baseline results if they exist
try:
    with open('results/baseline_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    fixed_revenue = baseline_results['Fixed Pricing']['total_revenue']
    rule_revenue = baseline_results['Rule-Based']['total_revenue']
    hist_revenue = baseline_results['Historical Average']['total_revenue']
    
    print("Comparison with baselines:")
    print(f"  Fixed Pricing:      ${fixed_revenue:>12,.0f}")
    print(f"  Rule-Based:         ${rule_revenue:>12,.0f}")
    print(f"  Historical Average: ${hist_revenue:>12,.0f}")
    print(f"  GA:                 ${scaled_revenue:>12,.0f}")
    print()
    
    ga_vs_fixed = ((scaled_revenue / fixed_revenue) - 1) * 100
    print(f"GA vs Fixed Pricing: {ga_vs_fixed:+.1f}%")
    
    if ga_vs_fixed > 10:
        print("✅ EXCELLENT: GA significantly outperforms baseline!")
    elif ga_vs_fixed > 0:
        print("✅ GOOD: GA outperforms baseline")
    elif ga_vs_fixed > -10:
        print("✓ ACCEPTABLE: GA comparable to baseline")
    else:
        print("⚠️ GA underperforms baseline")

except FileNotFoundError:
    print("ℹ️ Baseline results not found - run baselines first for comparison")
print()

# ============================================================================
# STEP 7: Save Results
# ============================================================================

print("="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

# Prepare results
ga_results = {
    'strategy_name': 'Genetic Algorithm (Fast)',
    'total_revenue': float(scaled_revenue),
    'test_revenue_raw': float(test_results['total_revenue']),
    'bookings_made': int(test_results['bookings_made']),
    'avg_price': float(test_results['avg_price']),
    'booking_rate': float(test_results['booking_rate']),
    'population_size': POPULATION_SIZE,
    'generations': GENERATIONS,
    'training_time_minutes': train_time / 60,
    'best_fitness': float(history['best_fitness'][-1]),
    'optimization': {
        'sample_size': SAMPLE_SIZE,
        'mode': 'FAST'
    }
}

# Save results
import os
os.makedirs('results', exist_ok=True)

with open('results/ga_results_fast.json', 'w') as f:
    json.dump(ga_results, f, indent=4)
print("✓ Results saved to results/ga_results_fast.json")

# Save GA model
ga.save('results/ga_agent_fast')
print("✓ Model saved to results/ga_agent_fast_ga.pkl")

# Save best strategy
np.save('results/ga_best_strategy_fast.npy', best_strategy)
print("✓ Best strategy saved to results/ga_best_strategy_fast.npy")

print()

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

print("="*80)
print("✅ FAST GA TRAINING COMPLETE!")
print("="*80)
print()
print("Summary:")
print(f"  Total Runtime:     {total_time/60:.1f} minutes")
print(f"  Population:        {POPULATION_SIZE}")
print(f"  Generations:       {GENERATIONS}")
print(f"  Sample Size:       {SAMPLE_SIZE:,}")
print(f"  Final Revenue:     ${scaled_revenue:,.0f}")
print(f"  Test Bookings:     {test_results['bookings_made']:,}")
print()
print("Files saved:")
print("  - results/ga_results_fast.json")
print("  - results/ga_agent_fast_ga.pkl")
print("  - results/ga_best_strategy_fast.npy")
print()
print("Next steps:")
print("  1. Compare DQN vs GA results")
print("  2. Create visualizations")
print("  3. Write final report")
print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
