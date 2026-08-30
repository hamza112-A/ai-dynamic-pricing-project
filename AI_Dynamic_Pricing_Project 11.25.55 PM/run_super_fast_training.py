
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime

# Import optimized modules
from environment_optimized import create_optimized_environment
from rl_agent_optimized import OptimizedDQNAgent
from evaluation import BaselineStrategy, PerformanceEvaluator

print("="*80)
print("SUPER FAST DQN TRAINING - EXTREME CPU OPTIMIZATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# SUPER FAST CONFIGURATION
# ============================================================================

DATA_PATH = 'data/processed_data.csv'
EPISODES = 200  # Max episodes
SAMPLE_SIZE = 50000  # 50K dataset (good balance)
TARGET_UPDATE = 5
EVAL_EPISODES = 50
EARLY_STOPPING_START = 75  # Don't stop before epsilon is low
EARLY_STOPPING_PATIENCE = 30

print("[CONFIG] Training Configuration (50K Dataset + Improved Early Stopping):")
print(f"  Episodes: up to {EPISODES}")
print(f"  Data sample size: {SAMPLE_SIZE:,} rows (42% of full dataset)")
print(f"  Target update frequency: every {TARGET_UPDATE} episodes")
print(f"  Evaluation episodes: {EVAL_EPISODES}")
print(f"  Early stopping: start after episode {EARLY_STOPPING_START}, patience={EARLY_STOPPING_PATIENCE}")
print(f"  Optimization: Best model + delayed early stopping (wait for exploration)")
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

start_time = time.time()

# Create optimized environments (much faster!)
train_env = create_optimized_environment(
    DATA_PATH,
    train=True,
    sample_size=SAMPLE_SIZE
)

test_env = create_optimized_environment(
    DATA_PATH,
    train=False,
    sample_size=SAMPLE_SIZE
)

load_time = time.time() - start_time
print(f"\n✓ Environments created in {load_time:.1f}s")
print()

# ============================================================================
# STEP 2: Create Agent with AGGRESSIVE hyperparameters
# ============================================================================

print("="*80)
print("STEP 2: CREATING SUPER OPTIMIZED DQN AGENT")
print("="*80)

agent = OptimizedDQNAgent(
    state_size=train_env.STATE_DIM,
    action_size=train_env.ACTION_DIM,
    lr=0.002,  # Lower LR for full dataset stability
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.975,  # Slower decay for more episodes
    batch_size=256,
    reward_scale=0.001
)

print("✓ Agent created with AGGRESSIVE hyperparameters")
print(f"  Learning rate: {agent.lr} (very high)")
print(f"  Batch size: {agent.batch_size} (very large)")
print(f"  Epsilon decay: {agent.epsilon_decay} (very fast)")
print()

# ============================================================================
# STEP 3: Train Agent
# ============================================================================

print("="*80)
print(f"STEP 3: TRAINING ON 50K DATASET (up to {EPISODES} EPISODES)")
print("="*80)

train_start = time.time()

history = agent.train(
    train_env,
    episodes=EPISODES,
    target_update=TARGET_UPDATE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_start=EARLY_STOPPING_START,
    verbose=True
)

train_time = time.time() - train_start

print(f"\n✓ Training complete in {train_time/60:.1f} minutes")
print(f"  Episodes: {EPISODES}")
print(f"  Final epsilon: {agent.epsilon:.4f}")
print(f"  Average reward (last 10): {np.mean(history['episode_rewards'][-10:]):,.0f}")
print()

# ============================================================================
# STEP 4: Evaluate Agent
# ============================================================================

print("="*80)
print(f"STEP 4: EVALUATION ({EVAL_EPISODES} EPISODES)")
print("="*80)

eval_start = time.time()

test_results = agent.evaluate(test_env, episodes=EVAL_EPISODES, verbose=True)

eval_time = time.time() - eval_start

print(f"\n✓ Evaluation complete in {eval_time:.1f}s")
print()

# ============================================================================
# STEP 5: Calculate Final Metrics
# ============================================================================

print("="*80)
print("STEP 5: CALCULATING FINAL METRICS")
print("="*80)

# Calculate total revenue/bookings (raw, not scaled)
mean_revenue_per_ep = test_results['mean_revenue']
mean_bookings_per_ep = test_results['mean_bookings']

# Raw results from evaluation
total_revenue_raw = mean_revenue_per_ep * EVAL_EPISODES
total_bookings_raw = int(mean_bookings_per_ep * EVAL_EPISODES)

# For fair comparison: revenue per booking
revenue_per_booking = mean_revenue_per_ep / mean_bookings_per_ep if mean_bookings_per_ep > 0 else 0

print(f"DQN Agent Performance:")
print(f"  Mean Revenue/Episode: ${mean_revenue_per_ep:>12,.2f}")
print(f"  Mean Bookings/Episode: {mean_bookings_per_ep:>11,.0f}")
print(f"  Revenue per Booking:   ${revenue_per_booking:>12,.2f}")
print(f"  Total (20 episodes):   ${total_revenue_raw:>12,.2f}")
print(f"  Training Time:         {train_time/60:>12.1f} min")
print()

# ============================================================================
# STEP 6: Compare with Baselines
# ============================================================================

print("="*80)
print("STEP 6: BASELINE COMPARISON")
print("="*80)

# Load baseline results if they exist
try:
    with open('results/baseline_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    fixed_revenue = baseline_results['Fixed Pricing']['total_revenue']
    rule_revenue = baseline_results['Rule-Based']['total_revenue']
    hist_revenue = baseline_results['Historical Average']['total_revenue']
    
    # Calculate per-booking revenue for fair comparison
    fixed_bookings = baseline_results['Fixed Pricing']['bookings_made']
    fixed_per_booking = fixed_revenue / fixed_bookings if fixed_bookings > 0 else 0
    
    print("Comparison (Revenue per Booking):")
    print(f"  Fixed Pricing:      ${fixed_per_booking:>12,.2f} per booking")
    print(f"  DQN Agent:          ${revenue_per_booking:>12,.2f} per booking")
    print()
    
    improvement = ((revenue_per_booking / fixed_per_booking) - 1) * 100 if fixed_per_booking > 0 else 0
    print(f"DQN vs Fixed Pricing: {improvement:+.1f}%")
    
    if improvement > 15:
        print("✅ EXCELLENT: DQN significantly outperforms baseline!")
    elif improvement > 5:
        print("✅ GOOD: DQN outperforms baseline")
    elif improvement > -5:
        print("✓ ACCEPTABLE: DQN comparable to baseline (good for fast training)")
    else:
        print("⚠️ Note: DQN underperforms but this is expected with extreme speed optimization")

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
dqn_results = {
    'strategy_name': 'DQN Agent (Super Fast)',
    'mean_revenue_per_episode': float(mean_revenue_per_ep),
    'mean_bookings_per_episode': float(mean_bookings_per_ep),
    'revenue_per_booking': float(revenue_per_booking),
    'total_revenue_20_episodes': float(total_revenue_raw),
    'total_bookings_20_episodes': int(total_bookings_raw),
    'training_episodes': EPISODES,
    'evaluation_episodes': EVAL_EPISODES,
    'training_time_minutes': train_time / 60,
    'final_epsilon': float(agent.epsilon),
    'std_episode_revenue': float(test_results['std_revenue']),
    'optimization': {
        'sample_size': SAMPLE_SIZE,
        'learning_rate': agent.lr,
        'batch_size': agent.batch_size,
        'epsilon_decay': agent.epsilon_decay,
        'reward_scale': agent.reward_scale,
        'mode': 'EXTREME_FAST'
    }
}

# Save results
import os
os.makedirs('results', exist_ok=True)

with open('results/dqn_results_super_fast.json', 'w') as f:
    json.dump(dqn_results, f, indent=4)
print("✓ Results saved to results/dqn_results_super_fast.json")

# Save model
agent.save('results/dqn_agent_super_fast')
print("✓ Model saved to results/dqn_agent_super_fast_optimized_dqn.pkl")

# Save training history
import pickle
with open('results/dqn_history_super_fast.pkl', 'wb') as f:
    pickle.dump(history, f)
print("✓ History saved to results/dqn_history_super_fast.pkl")

print()

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

print("="*80)
print("✅ SUPER FAST TRAINING COMPLETE!")
print("="*80)
print()
print("Summary:")
print(f"  Total Runtime:     {total_time/60:.1f} minutes")
print(f"  Training Episodes: {len(history['episode_rewards'])} (early stop or completed)")
print(f"  Sample Size:       {SAMPLE_SIZE:,} rows (42% of dataset)")
print(f"  Revenue/Episode:   ${mean_revenue_per_ep:,.0f}")
print(f"  Revenue/Booking:   ${revenue_per_booking:,.2f}")
print()
print("Improvements Applied:")
print(f"  ✓ Delayed early stopping (after episode {EARLY_STOPPING_START})")
print(f"  ✓ Best model checkpointing")
print(f"  ✓ Lower learning rate (0.002 for stability)")
print(f"  ✓ 50K samples (optimal balance of speed & accuracy)")
print()
print("Files saved:")
print("  - results/dqn_results_super_fast.json")
print("  - results/dqn_agent_super_fast_optimized_dqn.pkl")
print("  - results/dqn_history_super_fast.pkl")
print()
print("Next steps:")
print("  1. Review results in results/dqn_results_super_fast.json")
print("  2. Run: python run_ga_fast.py (genetic algorithm)")
print("  3. Compare all strategies")
print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
