

import subprocess
import time
from datetime import datetime

print("="*80)
print("COMPLETE FAST PIPELINE - ALL ALGORITHMS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()

# ============================================================================
# STEP 1: Preprocessing
# ============================================================================

print("="*80)
print("STEP 1: DATA PREPROCESSING")
print("="*80)

step_start = time.time()
result = subprocess.run(['python', 'preprocess_optimized.py'], capture_output=False)
if result.returncode != 0:
    print("❌ Preprocessing failed!")
    exit(1)
step_time = time.time() - step_start
print(f"\n✓ Preprocessing complete in {step_time:.1f}s\n")

# ============================================================================
# STEP 2: Baselines
# ============================================================================

print("="*80)
print("STEP 2: BASELINE STRATEGIES")
print("="*80)

step_start = time.time()
result = subprocess.run(['python', 'run_baselines.py'], capture_output=False)
if result.returncode != 0:
    print("⚠️ Baselines failed, continuing anyway...")
step_time = time.time() - step_start
print(f"\n✓ Baselines complete in {step_time:.1f}s\n")

# ============================================================================
# STEP 3: DQN Training
# ============================================================================

print("="*80)
print("STEP 3: DQN REINFORCEMENT LEARNING")
print("="*80)

step_start = time.time()
result = subprocess.run(['python', 'run_super_fast_training.py'], capture_output=False)
if result.returncode != 0:
    print("❌ DQN training failed!")
    exit(1)
step_time = time.time() - step_start
print(f"\n✓ DQN complete in {step_time/60:.1f} minutes\n")

# ============================================================================
# STEP 4: Genetic Algorithm
# ============================================================================

print("="*80)
print("STEP 4: GENETIC ALGORITHM")
print("="*80)

step_start = time.time()
result = subprocess.run(['python', 'run_ga_fast.py'], capture_output=False)
if result.returncode != 0:
    print("❌ GA training failed!")
    exit(1)
step_time = time.time() - step_start
print(f"\n✓ GA complete in {step_time/60:.1f} minutes\n")

# ============================================================================
# STEP 5: Summary
# ============================================================================

total_time = time.time() - start_time

print("="*80)
print("✅ COMPLETE PIPELINE FINISHED!")
print("="*80)
print()
print(f"Total Runtime: {total_time/60:.1f} minutes")
print()
print("Results saved in:")
print("  - results/baseline_results.json")
print("  - results/dqn_results_super_fast.json")
print("  - results/ga_results_fast.json")
print()
print("Models saved in:")
print("  - results/dqn_agent_super_fast_optimized_dqn.pkl")
print("  - results/ga_agent_fast_ga.pkl")
print()
print("Next steps:")
print("  1. Review all results in results/ folder")
print("  2. Run comparison script to create visualizations")
print("  3. Write your project report")
print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
