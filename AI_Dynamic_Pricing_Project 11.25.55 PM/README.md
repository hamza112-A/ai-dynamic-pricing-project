# AI-Driven Dynamic Pricing for Hotel Revenue Optimization

**Student Name:** Hamza Arshad  

## Project Overview

This project implements a dynamic pricing strategy for hotel revenue optimization using:
1. **Reinforcement Learning (Q-Learning with Deep Q-Network)**
2. **Genetic Algorithm (Evolutionary Optimization)**

The system learns optimal pricing policies based on demand patterns, seasonality, booking windows, and other features from the Hotel Booking Demand dataset.

## Dataset

- **Source:** Hotel Booking Demand (Kaggle)
- **Description:** Contains booking information from two hotels (Resort and City) with features including arrival dates, market segments, booking changes, and previous cancellations.
- **Size:** ~119,000 bookings
- **Target:** Optimize Average Daily Rate (ADR) to maximize revenue

## Project Structure

```
AI_Dynamic_Pricing_Project/
├── data/
│   ├── hotel_bookings.csv          # Raw dataset (119K bookings)
│   └── processed_data.csv          # Preprocessed dataset (50K samples)
├── src/
│   ├── __init__.py
│   ├── environment_optimized.py    # MDP Environment (CPU optimized)
│   ├── rl_agent_optimized.py       # DQN Agent with Priority Replay
│   ├── genetic_algorithm.py        # GA implementation
│   └── evaluation.py               # Baselines & performance comparison
├── results/
│   ├── figures/
│   │   ├── preprocessing_summary.png      # Data visualization
│   │   ├── strategy_comparison.png        # Results comparison
│   │   └── training_performance.png       # Training curves
│   ├── baseline_results.json              # Baseline metrics
│   ├── dqn_results_super_fast.json        # DQN metrics
│   ├── ga_results_fast.json               # GA metrics
│   ├── final_comparison.csv               # Summary comparison
│   ├── dqn_agent_super_fast_optimized_dqn.pkl    # Trained DQN model
│   ├── dqn_history_super_fast.pkl         # DQN training history
│   ├── ga_agent_fast_ga.pkl               # Trained GA model
│   └── ga_best_strategy_fast.npy          # GA best strategy
├── report/
│   ├── 25207938_Mohammed_Abdul_Razzak_Baig_Report.tex
│   └── 25207938_Mohammed_Abdul_Razzak_Baig_Report.pdf
├── preprocess_optimized.py         # Data preprocessing script
├── run_baselines.py                # Run baseline strategies
├── run_super_fast_training.py      # Train DQN (7-8 min)
├── run_ga_fast.py                  # Train GA (0.1 sec)
├── run_complete_fast.py            # Run full pipeline
├── create_training_performance_figure.py  # Generate training figure
├── requirements.txt
└── README.md
```

## Installation & Setup

### 1. Prerequisites
- Python 3.9 or higher
- pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torch (PyTorch for DQN)
- scipy

### 3. Dataset
The dataset (`data/hotel_bookings.csv`) is included in the project.
- **Source:** Hotel Booking Demand (Kaggle)
- **Size:** 119,323 bookings
- **Preprocessed:** 50,000 samples used for training (42% of full dataset)

## Running the Code

All scripts should be run from the project root directory:
```bash
cd AI_Dynamic_Pricing_Project
```

### Option 1: Run Complete Pipeline (Recommended)
Run everything in sequence: preprocessing, baselines, DQN, and GA.

```bash
python run_complete_fast.py
```

**Total time:** ~10-15 minutes on CPU  
**Output:** All results, models, and figures

---

### Option 2: Run Individual Steps

#### Step 1: Data Preprocessing
```bash
python preprocess_optimized.py
```
- **Time:** ~1 minute
- **Output:** `data/processed_data.csv`, `results/figures/preprocessing_summary.png`
- **Creates:** Stratified sample of 50K bookings with engineered features

#### Step 2: Baseline Strategies
```bash
python run_baselines.py
```
- **Time:** ~1 second
- **Output:** `results/baseline_results.json`, `results/figures/strategy_comparison.png`
- **Baselines:** Fixed pricing, rule-based, historical average

#### Step 3: Train DQN Agent
```bash
python run_super_fast_training.py
```
- **Time:** 7-8 minutes on CPU
- **Output:** 
  - `results/dqn_results_super_fast.json`
  - `results/dqn_agent_super_fast_optimized_dqn.pkl`
  - `results/dqn_history_super_fast.pkl`
- **Algorithm:** Deep Q-Network with Double DQN, Priority Replay, Early Stopping
- **Configuration:** 150 episodes (stops early at best performance)

#### Step 4: Train Genetic Algorithm
```bash
python run_ga_fast.py
```
- **Time:** 0.1 seconds
- **Output:**
  - `results/ga_results_fast.json`
  - `results/ga_agent_fast_ga.pkl`
  - `results/ga_best_strategy_fast.npy`
- **Algorithm:** Evolutionary optimization
- **Configuration:** Population 50, 30 generations

#### Step 5: Generate Training Performance Figure
```bash
python create_training_performance_figure.py
```
- **Time:** ~1 second
- **Output:** `results/figures/training_performance.png`
- **Shows:** DQN reward curve, epsilon decay, loss, and GA fitness evolution
- **Requires:** DQN and GA must be trained first

---

### View Results

All results are saved in JSON format:

```bash
# View baseline results
cat results/baseline_results.json

# View DQN results
cat results/dqn_results_super_fast.json

# View GA results
cat results/ga_results_fast.json

# View comparison table
cat results/final_comparison.csv
```

## Key Results

| Strategy | Revenue/Booking | Improvement | Training Time |
|----------|----------------|-------------|---------------|
| Fixed Pricing (Baseline) | $100.84 | 0.0% | 0s |
| Rule-Based | $122.96 | +21.9% | 0s |
| Historical Average | $98.95 | -1.9% | 0s |
| **DQN (Reinforcement Learning)** | **$130.15** | **+29.1%** | 7.7 min |
| **Genetic Algorithm** | **$145.05** | **+43.8%** | 0.1s |

**Key Findings:**
- Both AI methods substantially outperform traditional pricing strategies
- GA achieves highest revenue but requires full retraining for new scenarios
- DQN offers good performance with online learning capability
- Training is CPU-friendly (<10 minutes total)
- Results are stable (±5.7% coefficient of variation)

## Methodology

### Problem Formulation (MDP)

**State Space:**
- Lead time (days until arrival)
- Season (Spring, Summer, Fall, Winter)
- Day of week
- Market segment
- Historical demand (rolling average)
- Current occupancy rate

**Action Space:**
- Pricing multiplier: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

**Reward Function:**
```
reward = (price × booking_probability) - penalty_for_extreme_pricing
```

**Environment Dynamics:**
- Demand prediction model (Random Forest)
- Booking probability as function of price and features

### Algorithms Implemented

#### 1. Deep Q-Network (DQN) - Reinforcement Learning
**Architecture:**
- Neural network: 3 layers (64 → 32 neurons, ReLU activation)
- Input: 7 state features
- Output: 9 Q-values (one per action)

**Key Features:**
- Double DQN to prevent Q-value overestimation
- Priority Experience Replay (learn from important transitions)
- Early stopping with best model checkpointing
- Reward scaling (0.001×) to prevent gradient explosion
- Epsilon decay: 1.0 → 0.049 over 100 episodes

**Hyperparameters:**
- Learning rate: 0.005
- Batch size: 256
- Discount factor (γ): 0.95
- Epsilon decay: 0.96 per episode
- Memory size: 10,000 transitions
- Target network update: Every 3 episodes

**Training:**
- Episodes: 150 (with early stopping)
- Best model saved at episode 120
- Training time: 7.7 minutes on CPU

#### 2. Genetic Algorithm - Evolutionary Optimization
**Chromosome Representation:**
- 50-dimensional vector
- Each gene: pricing multiplier for specific scenario
- Scenarios: Discretized (lead time × month × day)

**Genetic Operators:**
- **Selection:** Tournament selection (k=3)
- **Crossover:** Two-point crossover (rate: 0.7)
- **Mutation:** Gaussian mutation (rate: 0.2, std: 0.1)

**Configuration:**
- Population size: 50 individuals
- Generations: 30
- Elitism: Top 10% preserved

**Fitness Function:**
```
fitness = total_revenue - penalty_for_extreme_prices
```

**Training:**
- Generations: 30
- Training time: 0.1 seconds

## Evaluation Metrics

1. **Total Revenue:** Sum of all booking revenues
2. **Average Revenue per Booking:** Mean ADR
3. **Occupancy Rate:** Percentage of available rooms booked
4. **Price Stability:** Standard deviation of prices
5. **Revenue per Available Room (RevPAR)**

## Technologies Used

- **Python 3.11**
- **PyTorch:** Deep learning for DQN
- **NumPy/Pandas:** Data manipulation and analysis
- **Scikit-learn:** Preprocessing and utilities
- **Matplotlib/Seaborn:** Visualization
- **SciPy:** Numerical operations

No Jupyter notebooks required - all code runs as standalone Python scripts.

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError`
- **Solution:** Install dependencies: `pip install -r requirements.txt`

**Issue:** `FileNotFoundError: hotel_bookings.csv`
- **Solution:** Ensure you're in the project root directory and data file exists

**Issue:** Training takes too long
- **Solution:** Pre-trained models are included in `results/` folder. You can use them directly or reduce training episodes in the scripts.

**Issue:** Out of memory error
- **Solution:** The code is optimized for CPU with 8GB RAM. If issues persist, reduce `SAMPLE_SIZE` in `run_super_fast_training.py`

**Issue:** Figures not generating
- **Solution:** Ensure matplotlib backend is installed: `pip install matplotlib pillow`

---## License

For academic and educational purposes only.

---

**Hamza Arshad**  
MSc in Artificial Intelligence  
