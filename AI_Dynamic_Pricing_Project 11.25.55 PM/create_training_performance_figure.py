

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("Loading training results...")

# Load DQN results
with open('results/dqn_results_super_fast.json', 'r') as f:
    dqn_results = json.load(f)

# Load DQN history
with open('results/dqn_history_super_fast.pkl', 'rb') as f:
    dqn_history = pickle.load(f)

# Load GA results  
with open('results/ga_results_fast.json', 'r') as f:
    ga_results = json.load(f)

# Load GA agent (contains history)
with open('results/ga_agent_fast_ga.pkl', 'rb') as f:
    ga_agent = pickle.load(f)

print("Creating combined figure...")

# Create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Performance: DQN & Genetic Algorithm', fontsize=16, fontweight='bold')

# ============================================================================
# Plot 1: DQN Reward Learning Curve
# ============================================================================
ax = axes[0, 0]
rewards = dqn_history['episode_rewards']
episodes = range(1, len(rewards) + 1)

ax.plot(episodes, rewards, alpha=0.4, color='steelblue', linewidth=1, label='Episode Reward')

# Add rolling average (window=10)
if len(rewards) >= 10:
    window = 10
    rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(rewards) + 1), rolling_avg, 
            color='darkblue', linewidth=2, label=f'{window}-Episode Avg')

ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Total Reward', fontsize=11)
ax.set_title('DQN Learning Curve', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Add final performance annotation
final_reward = rewards[-1]
ax.annotate(f'Final: {final_reward:,.0f}', 
            xy=(len(rewards), final_reward),
            xytext=(len(rewards)*0.7, max(rewards)*0.9),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# ============================================================================
# Plot 2: DQN Epsilon Decay
# ============================================================================
ax = axes[0, 1]
epsilons = dqn_history['epsilon']
episodes_eps = range(1, len(epsilons) + 1)

ax.plot(episodes_eps, epsilons, color='orangered', linewidth=2.5, label='Epsilon (ε)')
ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Min ε = 0.05')
ax.fill_between(episodes_eps, 0, epsilons, alpha=0.2, color='orangered')

ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
ax.set_title('DQN Exploration-Exploitation Trade-off', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# Add annotations
ax.annotate('Exploration', xy=(5, 0.9), fontsize=10, color='darkred', weight='bold')
ax.annotate('Exploitation', xy=(len(epsilons)*0.7, 0.15), fontsize=10, color='darkgreen', weight='bold')

# ============================================================================
# Plot 3: DQN Training Loss
# ============================================================================
ax = axes[1, 0]
losses = dqn_history['loss']
episodes_loss = range(1, len(losses) + 1)

ax.plot(episodes_loss, losses, color='purple', linewidth=2, alpha=0.7, label='Training Loss')

# Add smoothed loss if enough data points
if len(losses) >= 5:
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(losses, size=5)
    ax.plot(episodes_loss, smoothed, color='darkviolet', linewidth=2.5, 
            linestyle='--', label='Smoothed (5-ep)')

ax.set_xlabel('Episode', fontsize=11)
ax.set_ylabel('MSE Loss', fontsize=11)
ax.set_title('DQN Training Loss Convergence', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Add convergence indicator
final_loss = losses[-1]
ax.annotate(f'Final: {final_loss:.4f}', 
            xy=(len(losses), final_loss),
            xytext=(len(losses)*0.6, max(losses)*0.8),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# ============================================================================
# Plot 4: GA Fitness Evolution
# ============================================================================
ax = axes[1, 1]

# Extract GA history
if hasattr(ga_agent, 'history'):
    ga_history = ga_agent.history
    generations = range(1, len(ga_history['best_fitness']) + 1)
    
    ax.plot(generations, ga_history['best_fitness'], 
            color='green', linewidth=2.5, marker='o', markersize=4, label='Best Fitness')
    ax.plot(generations, ga_history['avg_fitness'], 
            color='blue', linewidth=2, linestyle='--', marker='s', markersize=3, 
            alpha=0.7, label='Average Fitness')
    
    # Fill between best and average
    ax.fill_between(generations, ga_history['avg_fitness'], ga_history['best_fitness'], 
                    alpha=0.15, color='green')
    
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Fitness (Total Revenue $)', fontsize=11)
    ax.set_title('Genetic Algorithm Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    # Add final performance annotation
    final_best = ga_history['best_fitness'][-1]
    ax.annotate(f'Best: ${final_best:,.0f}', 
                xy=(len(generations), final_best),
                xytext=(len(generations)*0.5, final_best*0.95),
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
else:
    ax.text(0.5, 0.5, 'GA History Not Available', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Genetic Algorithm Evolution', fontsize=12, fontweight='bold')

# ============================================================================
# Save Figure
# ============================================================================
plt.tight_layout()

output_path = 'results/figures/training_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: {output_path}")

# Also save as high-res version
output_path_hires = 'results/figures/training_performance_hires.png'
plt.savefig(output_path_hires, dpi=600, bbox_inches='tight')
print(f"✓ High-res figure saved: {output_path_hires}")

plt.close()

print("\n" + "="*80)
print("TRAINING PERFORMANCE FIGURE GENERATION COMPLETE")
print("="*80)
print("\nGenerated figures:")
print(f"  • {output_path} (300 DPI)")
print(f"  • {output_path_hires} (600 DPI)")
print("\nFigure contains:")
print("  1. DQN reward learning curve with rolling average")
print("  2. DQN epsilon decay (exploration → exploitation)")
print("  3. DQN training loss convergence")
print("  4. GA fitness evolution (best and average)")
print("="*80)
