import numpy as np
import pandas as pd

from config_loader import load_config
from optimzers.base_optimzer import BaseOptimzer
from optimzers.optimzer_result import OptimzerResult

CONFIG = load_config()
OPT_CFG = CONFIG["optimizer"]
LAG_CFG = OPT_CFG["lagrangian"]


class LagrangianOptimizer(BaseOptimzer):
    """
    Subgradient-based Lagrangian relaxation:
    - Relaxes slice capacity constraints with multipliers lambda_k.
    - Solves user-wise subproblems for assignment and PRB amounts.
    - Repairs infeasible capacity violations greedily.
    """

    def __init__(self) -> None:
        super().__init__()
        self.max_iters = int(LAG_CFG.get("max_iters", 200))
        self.initial_step = float(LAG_CFG.get("initial_step", 2.0))
        self.step_decay = float(LAG_CFG.get("step_decay", 0.985))
        self.min_step = float(LAG_CFG.get("min_step", 0.01))

    def optimize(self, users_df: pd.DataFrame) -> OptimzerResult:
        prepared = self._prepare_inputs(users_df)
        n_users = prepared["n_users"]
        n_slices = prepared["n_slices"]
        targets = prepared["targets"]
        weights = prepared["weights"]
        prices = prepared["prices"]
        capacities = prepared["capacities"]
        costs = prepared["costs"]
        efficiencies = prepared["efficiencies"]
        users = prepared["users"]

        lambdas = np.zeros(n_slices, dtype=float)
        step = self.initial_step
        best = None

        for _ in range(self.max_iters):
            x_relaxed = np.zeros((n_users, n_slices), dtype=float)
            p_relaxed = np.zeros((n_users, n_slices), dtype=float)

            for i in range(n_users):
                best_value = 0.0
                best_k = -1
                best_p = 0.0

                for k in range(n_slices):
                    a = efficiencies[k] / max(targets[i], 1e-9)
                    p_sat = 1.0 / max(a, 1e-9)
                    eff_cost = costs[k] + lambdas[k]
                    p_min = self.min_prbs_per_admitted_user

                    candidates = [p_min, p_sat]
                    candidate_values = []
                    for p in candidates:
                        sat = min(1.0, a * p)
                        value = prices[i] + weights[i] * sat - eff_cost * p
                        candidate_values.append((value, p))

                    local_value, local_p = max(candidate_values, key=lambda t: t[0])
                    if local_value > best_value:
                        best_value = local_value
                        best_k = k
                        best_p = local_p

                if best_k >= 0:
                    x_relaxed[i, best_k] = 1.0
                    p_relaxed[i, best_k] = best_p

            p_feasible = self._repair_capacities(x_relaxed.copy(), p_relaxed.copy(), prepared)
            x_feasible = (p_feasible > 0).astype(float)

            # Ensure at-most-one assignment by construction; this keeps robustness if repaired rows are all zeros.
            for i in range(n_users):
                if x_feasible[i].sum() > 1:
                    keep = int(np.argmax(p_feasible[i]))
                    x_feasible[i] = 0.0
                    x_feasible[i, keep] = 1.0

            p_feasible = self._enforce_min_tier_satisfaction(x_feasible, p_feasible, prepared)
            x_feasible = (p_feasible > 0).astype(float)

            result = self._build_result(
                prepared,
                x_sol=x_feasible,
                p_sol=p_feasible,
                status=0,
                success=True,
                message="Lagrangian heuristic completed.",
            )

            if best is None or result.objective_value > best.objective_value:
                best = result

            used = p_relaxed.sum(axis=0)
            violation = used - capacities
            lambdas = np.maximum(0.0, lambdas + step * violation)
            step = max(self.min_step, step * self.step_decay)

        if best is None:
            raise RuntimeError("Lagrangian optimizer failed to produce a solution.")

        return best

    def _repair_capacities(self, x_sol: np.ndarray, p_sol: np.ndarray, prepared: dict) -> np.ndarray:
        capacities = prepared["capacities"]
        costs = prepared["costs"]
        efficiencies = prepared["efficiencies"]
        targets = prepared["targets"]
        weights = prepared["weights"]
        prices = prepared["prices"]
        n_users, n_slices = p_sol.shape

        for k in range(n_slices):
            while p_sol[:, k].sum() > capacities[k] + 1e-9:
                active = np.where(p_sol[:, k] > 0)[0]
                if active.size == 0:
                    break

                penalties = []
                for i in active:
                    p = p_sol[i, k]
                    sat = min(1.0, efficiencies[k] * p / max(targets[i], 1e-9))
                    value = prices[i] + weights[i] * sat - costs[k] * p
                    penalties.append((value / max(p, 1e-9), i))

                _, drop_i = min(penalties, key=lambda t: t[0])
                p_sol[drop_i, k] = 0.0
                x_sol[drop_i, :] = 0.0

        return p_sol

    def _enforce_min_tier_satisfaction(
        self, x_sol: np.ndarray, p_sol: np.ndarray, prepared: dict
    ) -> np.ndarray:
        users = prepared["users"]
        targets = prepared["targets"]
        capacities = prepared["capacities"]
        efficiencies = prepared["efficiencies"]

        allocated_rate = np.sum(p_sol * efficiencies.reshape(1, -1), axis=1)
        sat = np.minimum(1.0, allocated_rate / np.maximum(targets, 1e-9))
        satisfied = sat >= self.threshold

        for tier_name in self.tier_names:
            tier_indices = users.index[users["Tier"] == tier_name].tolist()
            required = min(int(self.min_satisfied.get(tier_name, 0)), len(tier_indices))
            current = int(np.sum(satisfied[tier_indices])) if tier_indices else 0
            needed = required - current
            if needed <= 0:
                continue

            candidates = [i for i in tier_indices if not satisfied[i]]
            for i in candidates:
                if needed <= 0:
                    break
                # Keep existing assigned slice if present, otherwise choose most efficient one.
                if x_sol[i].sum() > 0:
                    k = int(np.argmax(x_sol[i]))
                else:
                    k = int(np.argmax(efficiencies))

                required_rate = self.threshold * targets[i]
                required_prbs = required_rate / max(efficiencies[k], 1e-9)
                additional = max(0.0, required_prbs - p_sol[i, k])
                available = capacities[k] - p_sol[:, k].sum()
                if available <= 0:
                    continue
                grant = min(additional, available)
                if grant <= 0:
                    continue

                p_sol[i, k] += grant
                x_sol[i, :] = 0.0
                x_sol[i, k] = 1.0

                new_rate = np.sum(p_sol[i] * efficiencies)
                satisfied[i] = (new_rate / max(targets[i], 1e-9)) >= self.threshold
                if satisfied[i]:
                    needed -= 1

        return p_sol
