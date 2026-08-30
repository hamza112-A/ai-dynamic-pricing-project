
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import json, os


# ---------------------------------------------------------------------------
# Baseline Strategies
# ---------------------------------------------------------------------------

class BaselineStrategy:
    """Three simple baseline pricing strategies for comparison."""

    @staticmethod
    def fixed_pricing(data: pd.DataFrame, base_price: float = 100.0) -> Dict:
        """Constant price equal to the mean ADR."""
        if "is_canceled" in data.columns:
            booked = (1 - data["is_canceled"].values) > 0.5
        else:
            booked = np.ones(len(data), dtype=bool)
        total = float(base_price * booked.sum())
        return {"total_revenue": total, "bookings_made": int(booked.sum()),
                "avg_price": base_price, "booking_rate": booked.mean(),
                "strategy_name": "Fixed Pricing"}

    @staticmethod
    def rule_based_pricing(data: pd.DataFrame, base_price: float = 100.0) -> Dict:
        """Hand-crafted rules: season, weekend, short lead-time."""
        mults = np.ones(len(data))

        months = data.get("arrival_month", pd.Series(np.full(len(data), 6))).values
        dow    = data.get("arrival_day_of_week", pd.Series(np.full(len(data), 3))).values
        lt     = data.get("lead_time", pd.Series(np.full(len(data), 30))).values

        mults[np.isin(months, [6, 7, 8])] *= 1.3
        mults[np.isin(dow, [5, 6])]       *= 1.2
        mults[lt < 7]                      *= 1.4
        mults = np.clip(mults, 0.7, 1.5)

        prices = base_price * mults

        if "is_canceled" in data.columns:
            base_prob = 1 - data["is_canceled"].values
        else:
            base_prob = np.full(len(data), 0.7)

        effect = np.exp(-2.0 * (mults - 1.0))
        prob   = np.clip(base_prob * (0.3 + 0.7 * effect), 0, 1)
        booked = prob > 0.5

        total = float((prices * booked).sum())
        return {"total_revenue": total, "bookings_made": int(booked.sum()),
                "avg_price": float(prices[booked].mean()) if booked.any() else 0,
                "booking_rate": float(booked.mean()),
                "strategy_name": "Rule-Based Pricing"}

    @staticmethod
    def historical_avg_pricing(data: pd.DataFrame, base_price: float = 100.0) -> Dict:
        """Monthly historical average prices."""
        if "adr" in data.columns and "arrival_month" in data.columns:
            monthly = data.groupby("arrival_month")["adr"].mean().to_dict()
        else:
            monthly = {i: base_price for i in range(1, 13)}

        months = data.get("arrival_month", pd.Series(np.full(len(data), 6))).values
        prices = np.array([monthly.get(m, base_price) for m in months])
        mults  = prices / base_price

        if "is_canceled" in data.columns:
            base_prob = 1 - data["is_canceled"].values
        else:
            base_prob = np.full(len(data), 0.7)

        effect = np.exp(-2.0 * (mults - 1.0))
        prob   = np.clip(base_prob * (0.3 + 0.7 * effect), 0, 1)
        booked = prob > 0.5

        total = float((prices * booked).sum())
        return {"total_revenue": total, "bookings_made": int(booked.sum()),
                "avg_price": float(prices[booked].mean()) if booked.any() else 0,
                "booking_rate": float(booked.mean()),
                "strategy_name": "Historical Avg Pricing"}


# ---------------------------------------------------------------------------
# Performance Evaluator
# ---------------------------------------------------------------------------

class PerformanceEvaluator:
    """Aggregate results from many strategies and produce comparisons."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def add_result(self, name: str, metrics: Dict):
        self.results[name] = metrics

    def compare(self, baseline: str = "Fixed Pricing") -> pd.DataFrame:
        if baseline not in self.results:
            baseline = next(iter(self.results))
        base_rev = self.results[baseline]["total_revenue"]
        rows = []
        for name, m in self.results.items():
            rev = m["total_revenue"]
            imp = (rev - base_rev) / base_rev * 100 if base_rev else 0
            rows.append({
                "Strategy": name,
                "Total Revenue ($)": f"{rev:,.0f}",
                "Improvement (%)": f"{imp:+.1f}",
                "Bookings": m.get("bookings_made", "N/A"),
                "Avg Price ($)": f"{m.get('avg_price', 0):.2f}",
                "Booking Rate": f"{m.get('booking_rate', 0):.1%}",
            })
        return pd.DataFrame(rows)

    def statistical_test(self, name1: str, revs1: List[float],
                         name2: str, revs2: List[float]) -> Dict:
        t, p = stats.ttest_ind(revs1, revs2)
        return {"strategies": (name1, name2), "t": t, "p": p,
                "significant": p < 0.05,
                "mean1": np.mean(revs1), "mean2": np.mean(revs2)}

    # ---- Visualisation ---------------------------------------------------

    def plot_comparison(self, save_dir: str = "results/figures"):
        """Generate a 2×2 comparison figure and save it."""
        os.makedirs(save_dir, exist_ok=True)
        if not self.results:
            return

        names = list(self.results.keys())
        revs  = [self.results[n]["total_revenue"] for n in names]
        books = [self.results[n].get("bookings_made", 0) for n in names]
        avg_p = [self.results[n].get("avg_price", 0) for n in names]

        palette = sns.color_palette("viridis", len(names))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Pricing Strategy Comparison", fontsize=16, fontweight="bold")

        # 1 — Revenue
        ax = axes[0, 0]
        bars = ax.bar(names, revs, color=palette)
        ax.set_title("Total Revenue"); ax.set_ylabel("Revenue ($)")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"${b.get_height():,.0f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        # 2 — Bookings
        ax = axes[0, 1]
        bars = ax.bar(names, books, color=palette)
        ax.set_title("Bookings Made"); ax.set_ylabel("Count")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        # 3 — Average Price
        ax = axes[1, 0]
        bars = ax.bar(names, avg_p, color=palette)
        ax.set_title("Average Price"); ax.set_ylabel("Price ($)")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"${b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        # 4 — Improvement %
        ax = axes[1, 1]
        base = revs[0] if revs[0] else 1
        imps = [(r - base) / base * 100 for r in revs]
        cols = ["#888" if i == 0 else ("#2ca02c" if v > 0 else "#d62728")
                for i, v in enumerate(imps)]
        bars = ax.bar(names, imps, color=cols)
        ax.set_title("Revenue Improvement vs Baseline"); ax.set_ylabel("%")
        ax.axhline(0, color="k", lw=0.8)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{b.get_height():+.1f}%", ha="center",
                    va="bottom" if b.get_height() >= 0 else "top", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        path = os.path.join(save_dir, "strategy_comparison.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {path}")

    def plot_learning_curve(self, rewards: list, title: str = "DQN Learning Curve",
                            save_dir: str = "results/figures"):
        """Plot RL training curve with rolling average."""
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
        # Rolling average
        window = min(50, len(rewards) // 5 + 1)
        if window > 1:
            avg = pd.Series(rewards).rolling(window).mean()
            ax.plot(avg, color="darkblue", linewidth=2, label=f"{window}-ep Moving Avg")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
        ax.legend(); ax.grid(alpha=0.3)
        path = os.path.join(save_dir, "dqn_learning_curve.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {path}")

    def plot_ga_evolution(self, history: dict, save_dir: str = "results/figures"):
        """Plot GA fitness evolution."""
        os.makedirs(save_dir, exist_ok=True)
        gens = range(1, len(history["best_fitness"]) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gens, history["best_fitness"],  "g-",  lw=2, label="Best")
        ax.plot(gens, history["avg_fitness"],   "b--", lw=1.5, label="Average")
        ax.plot(gens, history["worst_fitness"], "r:",  lw=1, label="Worst")
        ax.fill_between(gens, history["worst_fitness"], history["best_fitness"],
                        alpha=0.1, color="green")
        ax.set_title("GA Fitness Evolution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generation"); ax.set_ylabel("Fitness (Revenue $)")
        ax.legend(); ax.grid(alpha=0.3)
        path = os.path.join(save_dir, "ga_evolution.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {path}")

    def plot_price_distribution(self, prices_dict: Dict[str, list],
                                save_dir: str = "results/figures"):
        """KDE of price distributions across strategies."""
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, prices in prices_dict.items():
            sns.kdeplot(prices, ax=ax, label=name, fill=True, alpha=0.25)
        ax.set_title("Price Distribution by Strategy", fontsize=14, fontweight="bold")
        ax.set_xlabel("Price ($)"); ax.legend()
        path = os.path.join(save_dir, "price_distributions.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {path}")

    def generate_report(self, save_path: str | None = None) -> str:
        lines = ["=" * 70, "PRICING STRATEGY EVALUATION REPORT", "=" * 70, ""]
        for name, m in self.results.items():
            lines.append(f"{name}:")
            lines.append(f"  Revenue:  ${m['total_revenue']:,.0f}")
            lines.append(f"  Bookings: {m.get('bookings_made', 'N/A')}")
            lines.append(f"  Avg Price: ${m.get('avg_price', 0):.2f}")
            if "booking_rate" in m:
                lines.append(f"  Book Rate: {m['booking_rate']:.1%}")
            lines.append("")
        text = "\n".join(lines)
        print(text)
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                f.write(text)
        return text

    def save_results_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out = {}
        for name, m in self.results.items():
            out[name] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                         for k, v in m.items()
                         if not isinstance(v, (list, np.ndarray))}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[SAVE] Results → {path}")
