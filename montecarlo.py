import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from config_loader import load_config
from optimzers import LagrangianOptimizer, MilpOptimizer
from user_profile import User

CONFIG = load_config()


def _parse_users_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return [CONFIG["main"]["default_n_users"]]
    values = []
    for p in parts:
        try:
            values.append(int(p))
        except ValueError as exc:
            raise ValueError(f"Invalid user count: {p}") from exc
    return values


def _build_optimizer(method: str):
    method = method.strip().lower()
    if method == "milp":
        return MilpOptimizer()
    if method == "lagrangian":
        return LagrangianOptimizer()
    raise ValueError(f"Unknown optimizer method: {method}. Use 'milp' or 'lagrangian'.")


def run_montecarlo(
    n_users_list: List[int],
    n_runs: int,
    method: str,
    seed: int,
) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(seed)
    optimizer = _build_optimizer(method)

    for n_users in n_users_list:
        metrics = []
        for _ in range(n_runs):
            np.random.seed(int(rng.integers(0, 2**32 - 1)))
            users = User.generate_user_profile(n_users=n_users)
            result = optimizer.optimize(users)
            usage = result.slice_usage.set_index("Slice")
            metrics.append(
                {
                    "objective": result.objective_value,
                    "weighted_satisfaction": result.weighted_satisfaction,
                    "revenue": result.revenue,
                    "cost": result.cost,
                    "profit": result.profit,
                    "util_slice1": float(usage.loc["Slice1", "Utilization"]) * 100.0,
                    "util_slice2": float(usage.loc["Slice2", "Utilization"]) * 100.0,
                    "util_slice3": float(usage.loc["Slice3", "Utilization"]) * 100.0,
                }
            )

        df = pd.DataFrame(metrics)
        results.append(
            {
                "n_users": n_users,
                "n_runs": n_runs,
                "objective_mean": df["objective"].mean(),
                "objective_var": df["objective"].var(ddof=1),
                "objective_min": df["objective"].min(),
                "objective_max": df["objective"].max(),
                "weighted_satisfaction_mean": df["weighted_satisfaction"].mean(),
                "weighted_satisfaction_var": df["weighted_satisfaction"].var(ddof=1),
                "weighted_satisfaction_min": df["weighted_satisfaction"].min(),
                "weighted_satisfaction_max": df["weighted_satisfaction"].max(),
                "revenue_mean": df["revenue"].mean(),
                "revenue_var": df["revenue"].var(ddof=1),
                "revenue_min": df["revenue"].min(),
                "revenue_max": df["revenue"].max(),
                "cost_mean": df["cost"].mean(),
                "cost_var": df["cost"].var(ddof=1),
                "cost_min": df["cost"].min(),
                "cost_max": df["cost"].max(),
                "profit_mean": df["profit"].mean(),
                "profit_var": df["profit"].var(ddof=1),
                "profit_min": df["profit"].min(),
                "profit_max": df["profit"].max(),
                "util_slice1_mean": df["util_slice1"].mean(),
                "util_slice1_var": df["util_slice1"].var(ddof=1),
                "util_slice1_min": df["util_slice1"].min(),
                "util_slice1_max": df["util_slice1"].max(),
                "util_slice2_mean": df["util_slice2"].mean(),
                "util_slice2_var": df["util_slice2"].var(ddof=1),
                "util_slice2_min": df["util_slice2"].min(),
                "util_slice2_max": df["util_slice2"].max(),
                "util_slice3_mean": df["util_slice3"].mean(),
                "util_slice3_var": df["util_slice3"].var(ddof=1),
                "util_slice3_min": df["util_slice3"].min(),
                "util_slice3_max": df["util_slice3"].max(),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of optimizer metrics over multiple runs."
    )
    parser.add_argument(
        "--users-list",
        default=str(CONFIG["main"]["default_n_users"]),
        help="Comma-separated list of user counts (default from config.yaml).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of simulations per user count.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["milp", "lagrangian"],
        default=CONFIG["optimizer"].get("method", "milp").strip().lower(),
        help="Optimizer method to use (default from config.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["main"]["random_seed"],
        help="Seed for Monte Carlo runs (default from config.yaml).",
    )
    parser.add_argument(
        "--out",
        default="montecarlo_results.csv",
        help="Output CSV filename.",
    )
    args = parser.parse_args()

    user_counts = _parse_users_list(args.users_list)
    summary = run_montecarlo(
        n_users_list=user_counts,
        n_runs=args.runs,
        method=args.optimizer,
        seed=args.seed,
    )

    summary = summary.round(2)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".csv":
        out_path = results_dir / out_path.name
    else:
        out_path = results_dir / f"{out_path.name}.csv"

    summary.to_csv(out_path, index=False)
    print(summary)
    print(f"\nSaved results to: {out_path}")
