# This file is the main orchestrator.
import argparse
import numpy as np

from config_loader import load_config
from optimzers import LagrangianOptimizer, MilpOptimizer
from user_profile import User

CONFIG = load_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ISP slice allocation optimizer.")
    parser.add_argument(
        "--users",
        type=int,
        default=CONFIG["main"]["default_n_users"],
        help="Number of users to generate (default from config.yaml).",
    )
    parser.add_argument(
        "--optimizer",
        choices=["milp", "lagrangian"],
        default=CONFIG["optimizer"].get("method", "milp").strip().lower(),
        help="Optimizer method to use (default from config.yaml).",
    )
    args = parser.parse_args()

    np.random.seed(CONFIG["main"]["random_seed"])

    df_users = User.generate_user_profile(n_users=args.users)
    if args.optimizer == "milp":
        optimizer = MilpOptimizer()
    elif args.optimizer == "lagrangian":
        optimizer = LagrangianOptimizer()
    else:
        raise ValueError(
            f"Unknown optimizer method: {args.optimizer}. Use 'milp' or 'lagrangian'."
        )

    result = optimizer.optimize(df_users)

    print("=== Optimizer Status ===")
    print("Users generated:", len(df_users))
    print(f"Optimizer method: {args.optimizer.upper()}")
    print(f"success={result.success}, status={result.status}")
    print(result.message)
    print()
    print("=== Objective Breakdown ===")
    print(f"Objective: {result.objective_value:.4f}")
    print(f"Weighted Satisfaction: {result.weighted_satisfaction:.4f}")
    print(f"Revenue: {result.revenue:.4f}")
    print(f"Cost: {result.cost:.4f}")
    print(f"Profit: {result.profit:.4f}")
    print()
    print("=== Slice Usage ===")
    print(result.slice_usage)
    print()
    print("=== User Allocation Sample ===")
    print(result.user_allocations.head(10))
