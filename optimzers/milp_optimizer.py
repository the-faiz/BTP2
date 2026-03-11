import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from optimzers.base_optimzer import BaseOptimzer
from optimzers.optimzer_result import OptimzerResult


class MilpOptimizer(BaseOptimzer):
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

        x_start = 0
        x_size = n_users * n_slices
        p_start = x_start + x_size
        p_size = n_users * n_slices
        y_start = p_start + p_size
        y_size = n_users
        sat_start = y_start + y_size
        sat_size = n_users
        rate_start = sat_start + sat_size
        rate_size = n_users
        n_vars = rate_start + rate_size

        def x_idx(i: int, k: int) -> int:
            return x_start + i * n_slices + k

        def p_idx(i: int, k: int) -> int:
            return p_start + i * n_slices + k

        def y_idx(i: int) -> int:
            return y_start + i

        def sat_idx(i: int) -> int:
            return sat_start + i

        def rate_idx(i: int) -> int:
            return rate_start + i

        c = np.zeros(n_vars, dtype=float)
        for i in range(n_users):
            for k in range(n_slices):
                c[x_idx(i, k)] = -prices[i]
                c[p_idx(i, k)] = costs[k]
            c[sat_idx(i)] = -weights[i]

        lower_bounds = np.zeros(n_vars, dtype=float)
        upper_bounds = np.full(n_vars, np.inf, dtype=float)

        for i in range(n_users):
            for k in range(n_slices):
                upper_bounds[x_idx(i, k)] = 1.0
                upper_bounds[p_idx(i, k)] = capacities[k]
            upper_bounds[y_idx(i)] = 1.0
            upper_bounds[sat_idx(i)] = 1.0
            upper_bounds[rate_idx(i)] = float(np.dot(capacities, efficiencies))

        integrality = np.zeros(n_vars, dtype=int)
        integrality[x_start : x_start + x_size] = 1
        integrality[y_start : y_start + y_size] = 1

        constraints = []

        for i in range(n_users):
            row = np.zeros(n_vars, dtype=float)
            for k in range(n_slices):
                row[x_idx(i, k)] = 1.0
            constraints.append(LinearConstraint(row, -np.inf, 1.0))

        for k in range(n_slices):
            row = np.zeros(n_vars, dtype=float)
            for i in range(n_users):
                row[p_idx(i, k)] = 1.0
            constraints.append(LinearConstraint(row, -np.inf, capacities[k]))

        for i in range(n_users):
            for k in range(n_slices):
                row = np.zeros(n_vars, dtype=float)
                row[p_idx(i, k)] = 1.0
                row[x_idx(i, k)] = -capacities[k]
                constraints.append(LinearConstraint(row, -np.inf, 0.0))

        if self.min_prbs_per_admitted_user > 0:
            for i in range(n_users):
                row = np.zeros(n_vars, dtype=float)
                for k in range(n_slices):
                    row[p_idx(i, k)] = 1.0
                    row[x_idx(i, k)] -= self.min_prbs_per_admitted_user
                constraints.append(LinearConstraint(row, 0.0, np.inf))

        for i in range(n_users):
            row = np.zeros(n_vars, dtype=float)
            row[rate_idx(i)] = 1.0
            for k in range(n_slices):
                row[p_idx(i, k)] = -efficiencies[k]
            constraints.append(LinearConstraint(row, 0.0, 0.0))

        for i in range(n_users):
            row = np.zeros(n_vars, dtype=float)
            row[sat_idx(i)] = 1.0
            row[rate_idx(i)] = -1.0 / max(targets[i], 1e-9)
            constraints.append(LinearConstraint(row, -np.inf, 0.0))

        for i in range(n_users):
            row = np.zeros(n_vars, dtype=float)
            row[sat_idx(i)] = 1.0
            row[y_idx(i)] = -self.threshold
            constraints.append(LinearConstraint(row, 0.0, np.inf))

        upper_coeff = 1.0 - (self.threshold - self.eps)
        for i in range(n_users):
            row = np.zeros(n_vars, dtype=float)
            row[sat_idx(i)] = 1.0
            row[y_idx(i)] = -upper_coeff
            constraints.append(LinearConstraint(row, -np.inf, self.threshold - self.eps))

        for tier_name in self.tier_names:
            tier_indices = users.index[users["Tier"] == tier_name].tolist()
            min_required = int(self.min_satisfied.get(tier_name, 0))
            if min_required < 0:
                raise ValueError(
                    f"optimizer.minimum_satisfied_per_tier[{tier_name}] must be >= 0."
                )
            min_required = min(min_required, len(tier_indices))
            row = np.zeros(n_vars, dtype=float)
            for i in tier_indices:
                row[y_idx(i)] = 1.0
            constraints.append(LinearConstraint(row, float(min_required), np.inf))

        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
        )
        if result.x is None:
            raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")

        sol = result.x
        x_sol = np.array([sol[x_idx(i, k)] for i in range(n_users) for k in range(n_slices)]).reshape(
            n_users, n_slices
        )
        p_sol = np.array([sol[p_idx(i, k)] for i in range(n_users) for k in range(n_slices)]).reshape(
            n_users, n_slices
        )

        return self._build_result(
            prepared,
            x_sol=x_sol,
            p_sol=p_sol,
            status=int(result.status),
            success=bool(result.success),
            message=str(result.message),
        )
