

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")


def load_and_optimize_data(data_path: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Load data and optionally sample for faster training.
    
    Args:
        sample_size: If None, use full dataset. Otherwise sample N rows.
    """
    print(f"[DATA] Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Clean
    df = df[(df["adr"] >= 20) & (df["adr"] <= 500)].copy()
    
    if sample_size is not None and len(df) > sample_size:
        # Stratified sampling to keep data representative
        np.random.seed(seed)
        
        # Sample proportionally from different segments
        samples = []
        for month in df['arrival_month'].unique():
            month_data = df[df['arrival_month'] == month]
            n_samples = max(50, int(sample_size * len(month_data) / len(df)))
            if len(month_data) > n_samples:
                sampled = month_data.sample(n=n_samples, random_state=seed)
            else:
                sampled = month_data
            samples.append(sampled)
        
        df = pd.concat(samples, ignore_index=True)
        print(f"[DATA] Sampled {len(df):,} rows (from {sample_size:,} target)")
    else:
        print(f"[DATA] Using FULL dataset: {len(df):,} rows")
    
    print(f"[DATA] Data ready: {len(df):,} rows")
    return df


class OptimizedHotelPricingEnv:
    """
    Highly optimized pricing environment for FAST convergence.
    
    Changes from original:
    1. Better reward shaping (profit + occupancy balance)
    2. Richer state space (7 features vs 6)
    3. Pre-computed features (no repeated calculations)
    4. Curriculum learning (progressive difficulty)
    """
    
    PRICE_MULTIPLIERS = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    STATE_DIM = 7  # Increased from 6
    ACTION_DIM = 9
    
    def __init__(self, data: pd.DataFrame, base_price: float = 100.0,
                 max_steps: int = 1000, curriculum_level: float = 0.0):
        self.data = data.reset_index(drop=True)
        self.base_price = base_price
        self.max_steps = max_steps
        self.curriculum_level = curriculum_level  # 0.0=easy, 1.0=hard
        
        # Pre-compute all features for speed
        self._precompute_features()
        
        # Scaler for normalization
        self.scaler = StandardScaler()
        self.scaler.fit(self.precomputed_states)
        
        # Episode tracking
        self.reset()
    
    def _precompute_features(self):
        """
        Pre-compute all state features once.
        
        OPTIMIZATION: Computing features during step() is slow.
        We compute everything once and just index during episodes.
        """
        n = len(self.data)
        
        # Basic features
        lead_time_norm = self.data['lead_time'].values / 365.0
        season_norm = self.data['arrival_month'].values / 12.0
        dow_norm = self.data['arrival_day_of_week'].values / 7.0
        weekend = self.data['is_weekend'].values.astype(float)
        market_seg_norm = self.data['market_segment_encoded'].values / 5.0
        
        # Demand features (rolling average)
        is_booked = (1 - self.data['is_canceled'].values).astype(float)
        demand = np.convolve(is_booked, np.ones(50)/50, mode='same')
        
        # NEW: Price sensitivity feature
        # High lead time + high cancellations = price sensitive
        price_sensitivity = (
            (self.data['lead_time'].values / 365.0) * 0.5 +
            (self.data['previous_cancellations'].values / 5.0) * 0.5
        )
        price_sensitivity = np.clip(price_sensitivity, 0, 1)
        
        # Stack all features
        self.precomputed_states = np.column_stack([
            lead_time_norm,
            season_norm,
            dow_norm,
            weekend,
            market_seg_norm,
            demand,
            price_sensitivity  # NEW
        ]).astype(np.float32)
        
        # Pre-compute booking probabilities
        self.base_booking_probs = is_booked
        
        print(f"[ENV] Pre-computed {n:,} state vectors")
    
    def _get_booking_probability(self, idx: int, multiplier: float) -> float:
        """
        Calculate booking probability with BETTER price sensitivity.
        
        OPTIMIZATION: Uses pre-computed base probs + smart price effect
        """
        base_prob = self.base_booking_probs[idx]
        price_sensitivity = self.precomputed_states[idx, 6]  # NEW feature
        
        # More realistic price effect
        if multiplier < 1.0:
            # Lower price → higher booking (diminishing returns)
            price_effect = 1.0 + (1.0 - multiplier) * 0.5
        else:
            # Higher price → lower booking (elastic demand)
            price_effect = np.exp(-price_sensitivity * 3.0 * (multiplier - 1.0))
        
        prob = base_prob * price_effect
        
        # Curriculum learning: easier early on
        if self.curriculum_level < 0.5:
            # Make booking more predictable early in training
            prob = prob * 0.7 + 0.3  # Compress to [0.3, 1.0]
        
        return float(np.clip(prob, 0.01, 0.99))
    
    def _calculate_reward(self, revenue: float, booked: bool, multiplier: float) -> float:
        """
        OPTIMIZED reward function for faster learning.
        
        KEY CHANGES:
        1. Larger rewards (easier to learn)
        2. Balanced profit vs occupancy
        3. Bonus for good pricing
        4. Penalty for extremes
        """
        if not booked:
            # Penalty for losing booking
            return -50.0  # Significant penalty
        
        # Base reward is revenue (scaled up for learning)
        reward = revenue * 2.0  # Scale up
        
        # Bonus for optimal pricing (1.0-1.2 range)
        if 1.0 <= multiplier <= 1.2:
            reward += 30.0  # Encourage good pricing
        
        # Penalty for extreme pricing
        if multiplier < 0.8:
            reward -= 40.0  # Too cheap
        elif multiplier > 1.4:
            reward -= 40.0  # Too expensive
        
        return float(reward)
    
    def reset(self) -> np.ndarray:
        """Reset environment to start new episode."""
        self.current_step = 0
        self.current_index = np.random.randint(0, len(self.data))
        self.total_revenue = 0.0
        self.bookings_made = 0
        self.episode_history = []
        
        # Get normalized state
        state = self.precomputed_states[self.current_index]
        return self.scaler.transform(state.reshape(1, -1))[0].astype(np.float32)
    
    def step(self, action: int):
        """
        Execute one pricing decision.
        
        Returns: (next_state, reward, done, info)
        """
        multiplier = self.PRICE_MULTIPLIERS[action]
        price = self.base_price * multiplier
        
        # Get booking probability and outcome
        prob = self._get_booking_probability(self.current_index, multiplier)
        booked = np.random.random() < prob
        
        # Calculate revenue and reward
        revenue = price if booked else 0.0
        reward = self._calculate_reward(revenue, booked, multiplier)
        
        # Update episode stats
        if booked:
            self.bookings_made += 1
        self.total_revenue += revenue
        
        # Track for analysis
        self.episode_history.append({
            'step': self.current_step,
            'action': action,
            'multiplier': multiplier,
            'price': price,
            'booked': booked,
            'revenue': revenue,
            'reward': reward
        })
        
        # Move to next state
        self.current_step += 1
        self.current_index = (self.current_index + 1) % len(self.data)
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self.precomputed_states[self.current_index]
        next_state = self.scaler.transform(next_state.reshape(1, -1))[0].astype(np.float32)
        
        info = {
            'revenue': revenue,
            'booking_made': booked,
            'price': price,
            'booking_probability': prob,
            'total_revenue': self.total_revenue,
            'bookings_made': self.bookings_made
        }
        
        return next_state, reward, done, info
    
    def increase_difficulty(self):
        """Increase curriculum difficulty (for curriculum learning)."""
        self.curriculum_level = min(1.0, self.curriculum_level + 0.1)


def create_optimized_environment(data_path: str, train: bool = True,
                                train_split: float = 0.7,
                                sample_size: int = 10000) -> OptimizedHotelPricingEnv:
    """
    Factory function to create optimized environment.
    
    OPTIMIZATIONS:
    - Samples data intelligently (10K vs 119K rows)
    - Pre-computes features
    - Uses curriculum learning for training env
    """
    # Load and optimize data
    data = load_and_optimize_data(data_path, sample_size=sample_size)
    
    # Split train/test
    split_idx = int(len(data) * train_split)
    subset = data.iloc[:split_idx] if train else data.iloc[split_idx:]
    
    # Base price
    base_price = float(data['adr'].mean())
    
    # Create environment
    curriculum_level = 0.0 if train else 1.0  # Easy for training, hard for testing
    env = OptimizedHotelPricingEnv(
        subset.reset_index(drop=True),
        base_price=base_price,
        max_steps=1000,
        curriculum_level=curriculum_level
    )
    
    print(f"[ENV] {'Training' if train else 'Test'} environment created")
    print(f"  Data size: {len(subset):,} rows")
    print(f"  Base price: ${base_price:.2f}")
    print(f"  Curriculum level: {curriculum_level:.1f}")
    
    return env


# Test
if __name__ == "__main__":
    print("Testing Optimized Environment...")
    
    # Generate small test data
    test_data = pd.DataFrame({
        'lead_time': np.random.randint(0, 365, 1000),
        'arrival_month': np.random.randint(1, 13, 1000),
        'arrival_day_of_week': np.random.randint(0, 7, 1000),
        'is_weekend': np.random.randint(0, 2, 1000),
        'market_segment_encoded': np.random.randint(0, 5, 1000),
        'is_canceled': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'previous_cancellations': np.random.randint(0, 3, 1000),
        'adr': np.random.uniform(50, 200, 1000)
    })
    
    env = OptimizedHotelPricingEnv(test_data, base_price=100)
    state = env.reset()
    
    print(f"State shape: {state.shape} (should be {env.STATE_DIM})")
    
    total_reward = 0
    for i in range(10):
        action = np.random.randint(0, env.ACTION_DIM)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i}: action={action}, mult={env.PRICE_MULTIPLIERS[action]:.1f}, "
              f"reward={reward:.1f}, booked={info['booking_made']}")
    
    print(f"Total reward: {total_reward:.1f}")
    print("✓ Optimized Environment OK")
