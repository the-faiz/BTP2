import numpy as np
import pandas as pd

from config_loader import load_config
from optimzers.base_optimzer import BaseOptimzer
from optimzers.optimzer_result import OptimzerResult

CONFIG = load_config()
OPT_CFG = CONFIG["optimizer"]
PSO_CFG = OPT_CFG.get("pso", {})


class PsoOptimizer(BaseOptimzer):
    def __init__(self) -> None:
        super().__init__()
        self.n_particles = int(PSO_CFG.get("n_particles", 30))
        self.max_iters = int(PSO_CFG.get("max_iters", 200))
        self.w = float(PSO_CFG.get("w", 0.7))
        self.c1 = float(PSO_CFG.get("c1", 1.5))
        self.c2 = float(PSO_CFG.get("c2", 1.5))
        self.vmax = float(PSO_CFG.get("vmax", 10.0))

    def optimize(self, users_df: pd.DataFrame) -> OptimzerResult:
        prepared = self._prepare_inputs(users_df)
        n_users = prepared["n_users"]
        n_slices = prepared["n_slices"]
        capacities = prepared["capacities"]

        dim = n_users * n_slices
        ub = np.tile(capacities, n_users)
        lb = np.zeros(dim, dtype=float)

        rng = np.random.default_rng()
        positions = rng.uniform(lb, ub, size=(self.n_particles, dim))
        velocities = rng.uniform(-self.vmax, self.vmax, size=(self.n_particles, dim))

        pbest_pos = positions.copy()
        pbest_score = np.full(self.n_particles, -np.inf)
        gbest_pos = None
        gbest_score = -np.inf

        for _ in range(self.max_iters):
            for i in range(self.n_particles):
                score = self._evaluate_particle(positions[i], prepared)
                if score > pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i] = positions[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_pos = positions[i].copy()

            if gbest_pos is None:
                gbest_pos = positions[0].copy()

            r1 = rng.random(size=(self.n_particles, dim))
            r2 = rng.random(size=(self.n_particles, dim))
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            velocities = np.clip(velocities, -self.vmax, self.vmax)
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

        best_score = -np.inf
        best_result = None
        for i in range(self.n_particles):
            result = self._build_result_from_particle(positions[i], prepared)
            if result.objective_value > best_score:
                best_score = result.objective_value
                best_result = result

        if best_result is None:
            raise RuntimeError("PSO optimizer failed to produce a solution.")

        return best_result

    def _evaluate_particle(self, position: np.ndarray, prepared: dict) -> float:
        result = self._build_result_from_particle(position, prepared)
        return result.objective_value

    def _build_result_from_particle(
        self, position: np.ndarray, prepared: dict
    ) -> OptimzerResult:
        n_users = prepared["n_users"]
        n_slices = prepared["n_slices"]

        p_sol = position.reshape(n_users, n_slices).copy()
        x_sol, p_sol = self._assign_and_repair(p_sol, prepared)
        min_satisfied_violation = self._compute_min_satisfied_violation(
            x_sol, p_sol, prepared
        )

        return self._build_result(
            prepared,
            x_sol=x_sol,
            p_sol=p_sol,
            status=0,
            success=True,
            message="PSO heuristic completed.",
            min_satisfied_violation=min_satisfied_violation,
        )

    def _assign_and_repair(
        self, p_sol: np.ndarray, prepared: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = prepared["targets"]
        weights = prepared["weights"]
        prices = prepared["prices"]
        capacities = prepared["capacities"]
        costs = prepared["costs"]
        efficiencies = prepared["efficiencies"]
        n_users, n_slices = p_sol.shape

        x_sol = np.zeros_like(p_sol)
        for i in range(n_users):
            best_value = 0.0
            best_k = -1
            best_p = 0.0
            for k in range(n_slices):
                p = p_sol[i, k]
                if p <= 0:
                    continue
                sat = min(1.0, efficiencies[k] * p / max(targets[i], 1e-9))
                value = prices[i] + weights[i] * sat - costs[k] * p
                if value > best_value:
                    best_value = value
                    best_k = k
                    best_p = p

            p_sol[i, :] = 0.0
            if best_k >= 0 and best_p >= self.min_prbs_per_admitted_user:
                p_sol[i, best_k] = best_p
                x_sol[i, best_k] = 1.0

        for k in range(n_slices):
            total = p_sol[:, k].sum()
            if total > capacities[k] + 1e-9:
                scale = capacities[k] / total
                p_sol[:, k] *= scale

        for i in range(n_users):
            if x_sol[i].sum() == 0:
                continue
            k = int(np.argmax(x_sol[i]))
            if p_sol[i, k] < self.min_prbs_per_admitted_user:
                p_sol[i, :] = 0.0
                x_sol[i, :] = 0.0

        return x_sol, p_sol

    def _compute_min_satisfied_violation(
        self, x_sol: np.ndarray, p_sol: np.ndarray, prepared: dict
    ) -> float:
        users = prepared["users"]
        targets = prepared["targets"]
        efficiencies = prepared["efficiencies"]

        allocated_rate = np.sum(p_sol * efficiencies.reshape(1, -1), axis=1)
        sat = np.minimum(1.0, allocated_rate / np.maximum(targets, 1e-9))
        satisfied = sat >= self.threshold

        total_violation = 0.0
        for tier_name in self.tier_names:
            tier_indices = users.index[users["Tier"] == tier_name].tolist()
            min_required_raw = float(self.min_satisfied.get(tier_name, 0))
            if 0.0 < min_required_raw < 1.0:
                required = int(np.ceil(min_required_raw * len(tier_indices)))
            else:
                required = int(min_required_raw)
            required = min(required, len(tier_indices))
            current = int(np.sum(satisfied[tier_indices])) if tier_indices else 0
            total_violation += max(0, required - current)

        return float(total_violation)
