

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle, os


class GeneticAlgorithmPricing:
    """Genetic Algorithm for optimising hotel pricing strategy."""

    def __init__(self, data: pd.DataFrame, base_price: float = 100.0, *,
                 pop_size: int = 100, generations: int = 50,
                 crossover_prob: float = 0.7, mutation_prob: float = 0.2,
                 mutation_sigma: float = 0.1, tournament_k: int = 3,
                 n_scenarios: int = 50, seed: int = 42):
        self.data           = data.reset_index(drop=True)
        self.base_price     = base_price
        self.pop_size       = pop_size
        self.generations    = generations
        self.cx_prob        = crossover_prob
        self.mut_prob       = mutation_prob
        self.mut_sigma      = mutation_sigma
        self.tourn_k        = tournament_k
        self.n_scenarios    = n_scenarios
        self.rng            = np.random.RandomState(seed)

        # Pre-compute scenario indices for every row (avoids per-row dict access)
        self._scenario_idx = self._precompute_scenarios()

        # Pre-compute base booking probabilities
        self._base_probs = self._precompute_probs()

        # History
        self.history: Dict[str, list] = {
            "best_fitness": [], "avg_fitness": [], "worst_fitness": [],
            "best_individual": None,
        }

    # ---- pre-computation -------------------------------------------------

    def _precompute_scenarios(self) -> np.ndarray:
        lt  = np.clip(self.data.get("lead_time", pd.Series(np.full(len(self.data), 30))).values / 365.0, 0, 1)
        mon = self.data.get("arrival_month", pd.Series(np.full(len(self.data), 6))).values / 12.0
        dow = self.data.get("arrival_day_of_week", pd.Series(np.full(len(self.data), 3))).values / 7.0
        vals = lt * 0.5 + mon * 0.3 + dow * 0.2
        idx  = (vals * self.n_scenarios).astype(int)
        return np.clip(idx, 0, self.n_scenarios - 1)

    def _precompute_probs(self) -> np.ndarray:
        if "is_canceled" in self.data.columns:
            return (1 - self.data["is_canceled"].values).astype(np.float64)
        return np.full(len(self.data), 0.7)

    # ---- genetic operators -----------------------------------------------

    def _init_population(self) -> np.ndarray:
        return self.rng.uniform(0.7, 1.5, (self.pop_size, self.n_scenarios))

    def _fitness_vectorised(self, population: np.ndarray) -> np.ndarray:
        """Evaluate entire population at once (fast)."""
        # shape: (pop, n_rows)
        mults = population[:, self._scenario_idx]  # (pop, n_rows)
        prices = self.base_price * mults

        # Booking probability: base * (0.3 + 0.7 * exp(-2*(m-1)))
        price_effect = np.exp(-2.0 * (mults - 1.0))
        probs = self._base_probs[None, :] * (0.3 + 0.7 * price_effect)
        probs = np.clip(probs, 0, 1)

        # Deterministic booking (threshold 0.5)
        booked = (probs > 0.5).astype(np.float64)
        revenue = (prices * booked).sum(axis=1)

        # Penalty for extreme prices
        extreme = ((population < 0.75) | (population > 1.35)).sum(axis=1)
        return revenue - extreme * 100.0

    def _tournament_select(self, pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
        selected = np.empty_like(pop)
        for i in range(len(pop)):
            idxs = self.rng.choice(len(pop), self.tourn_k, replace=False)
            winner = idxs[np.argmax(fits[idxs])]
            selected[i] = pop[winner]
        return selected

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        if self.rng.random() > self.cx_prob:
            return p1.copy(), p2.copy()
        a, b = sorted(self.rng.choice(self.n_scenarios, 2, replace=False))
        c1, c2 = p1.copy(), p2.copy()
        c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
        return c1, c2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.n_scenarios) < self.mut_prob
        ind[mask] += self.rng.normal(0, self.mut_sigma, mask.sum())
        return np.clip(ind, 0.7, 1.5)

    # ---- main loop -------------------------------------------------------

    def evolve(self, verbose: bool = True) -> Tuple[np.ndarray, Dict]:
        pop  = self._init_population()
        fits = self._fitness_vectorised(pop)

        if verbose:
            print("=" * 60)
            print("GENETIC ALGORITHM OPTIMISATION")
            print(f"  Population={self.pop_size}  Generations={self.generations}")
            print(f"  Crossover={self.cx_prob}  Mutation={self.mut_prob}")
            print("=" * 60)

        for gen in range(1, self.generations + 1):
            # Selection
            parents = self._tournament_select(pop, fits)

            # Crossover + Mutation
            offspring = np.empty_like(parents)
            for i in range(0, self.pop_size, 2):
                j = min(i + 1, self.pop_size - 1)
                c1, c2 = self._crossover(parents[i], parents[j])
                offspring[i]  = self._mutate(c1)
                offspring[j]  = self._mutate(c2)

            # Evaluate
            off_fits = self._fitness_vectorised(offspring)

            # Elitism: keep best from old population
            best_old = np.argmax(fits)
            worst_new = np.argmin(off_fits)
            if fits[best_old] > off_fits[worst_new]:
                offspring[worst_new] = pop[best_old]
                off_fits[worst_new]  = fits[best_old]

            pop, fits = offspring, off_fits

            self.history["best_fitness"].append(float(fits.max()))
            self.history["avg_fitness"].append(float(fits.mean()))
            self.history["worst_fitness"].append(float(fits.min()))

            if verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}/{self.generations}  "
                      f"best=${fits.max():,.0f}  avg=${fits.mean():,.0f}")

        best = pop[np.argmax(fits)]
        self.history["best_individual"] = best.tolist()

        if verbose:
            print("=" * 60)
            print(f"  DONE — best revenue = ${fits.max():,.0f}")
            print("=" * 60)

        return best, self.history

    # ---- evaluation ------------------------------------------------------

    def evaluate_strategy(self, strategy: np.ndarray, verbose: bool = True) -> Dict:
        mults  = strategy[self._scenario_idx]
        prices = self.base_price * mults

        price_effect = np.exp(-2.0 * (mults - 1.0))
        probs  = self._base_probs * (0.3 + 0.7 * price_effect)
        probs  = np.clip(probs, 0, 1)
        booked = probs > 0.5

        total_revenue = float((prices * booked).sum())
        n_bookings    = int(booked.sum())

        results = {
            "total_revenue": total_revenue,
            "bookings_made": n_bookings,
            "booking_rate":  n_bookings / len(self.data),
            "avg_price":     float(prices[booked].mean()) if n_bookings else 0.0,
            "std_price":     float(prices[booked].std())  if n_bookings else 0.0,
            "min_price":     float(prices.min()),
            "max_price":     float(prices.max()),
        }
        if verbose:
            print(f"\n{'='*50}")
            print("GA STRATEGY EVALUATION")
            print(f"  Revenue:    ${results['total_revenue']:,.0f}")
            print(f"  Bookings:   {results['bookings_made']:,} "
                  f"({results['booking_rate']:.1%})")
            print(f"  Avg price:  ${results['avg_price']:.2f}")
            print(f"{'='*50}")
        return results

    # ---- persistence -----------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + "_ga.pkl", "wb") as f:
            pickle.dump({"history": self.history, "base_price": self.base_price,
                         "n_scenarios": self.n_scenarios}, f)

    def load(self, path: str):
        with open(path + "_ga.pkl", "rb") as f:
            d = pickle.load(f)
        self.history     = d["history"]
        self.base_price  = d["base_price"]
        self.n_scenarios = d["n_scenarios"]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from environment import generate_synthetic_data
    print("GA quick test …")
    df = generate_synthetic_data(2000)
    ga = GeneticAlgorithmPricing(df, base_price=100, pop_size=30, generations=10)
    best, hist = ga.evolve(verbose=True)
    ga.evaluate_strategy(best)
    print("✓ GA OK")
