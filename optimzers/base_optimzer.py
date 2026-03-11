from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from config_loader import load_config
from isp_model import ISP
from optimzers.optimzer_result import OptimzerResult

CONFIG = load_config()
OPT_CFG = CONFIG["optimizer"]


class BaseOptimzer(ABC):
    def __init__(self) -> None:
        self.isp = ISP()
        self.tiers_cfg = CONFIG["tiers"]
        self.threshold = float(OPT_CFG["satisfaction_threshold"])
        self.eps = float(OPT_CFG["satisfaction_link_eps"])
        self.min_prbs_per_admitted_user = float(OPT_CFG["min_prbs_per_admitted_user"])
        self.min_satisfied = OPT_CFG["minimum_satisfied_per_tier"]
        self.slice_names = list(self.isp.slices.keys())
        self.tier_names = list(self.tiers_cfg.keys())

        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("optimizer.satisfaction_threshold must be in (0, 1].")
        if not 0.0 <= self.eps < self.threshold:
            raise ValueError("optimizer.satisfaction_link_eps must be in [0, threshold).")
        if self.min_prbs_per_admitted_user < 0.0:
            raise ValueError("optimizer.min_prbs_per_admitted_user must be >= 0.")

    def _prepare_inputs(self, users_df: pd.DataFrame) -> dict:
        users = users_df.reset_index(drop=True).copy()
        if users.empty:
            raise ValueError("users_df is empty; cannot optimize allocation.")

        targets = users["Tier"].map(
            lambda tier: float(self.tiers_cfg[tier]["target_rate_mbps"])
        ).to_numpy()
        weights = users["Tier"].map(lambda tier: float(self.tiers_cfg[tier]["weight"])).to_numpy()
        prices = users["Tier"].map(lambda tier: float(self.tiers_cfg[tier]["price"])).to_numpy()
        capacities = np.array(
            [self.isp.capacity_prbs(name) for name in self.slice_names], dtype=float
        )
        costs = np.array([self.isp.cost_per_prb(name) for name in self.slice_names], dtype=float)
        efficiencies = np.array(
            [self.isp.efficiency_mhz_per_prb(name) for name in self.slice_names], dtype=float
        )

        return {
            "users": users,
            "targets": targets,
            "weights": weights,
            "prices": prices,
            "capacities": capacities,
            "costs": costs,
            "efficiencies": efficiencies,
            "n_users": len(users),
            "n_slices": len(self.slice_names),
        }

    def _build_result(
        self,
        prepared: dict,
        x_sol: np.ndarray,
        p_sol: np.ndarray,
        *,
        status: int,
        success: bool,
        message: str,
    ) -> OptimzerResult:
        users = prepared["users"].copy()
        targets = prepared["targets"]
        weights = prepared["weights"]
        prices = prepared["prices"]
        capacities = prepared["capacities"]
        costs = prepared["costs"]
        efficiencies = prepared["efficiencies"]

        allocated_rate = np.sum(p_sol * efficiencies.reshape(1, -1), axis=1)
        satisfaction = np.minimum(1.0, allocated_rate / np.maximum(targets, 1e-9))
        satisfied_flag = (satisfaction >= self.threshold).astype(int)

        assigned_slice_idx = np.where(x_sol.max(axis=1) > 0.5, x_sol.argmax(axis=1), -1)
        assigned_slices = [
            self.slice_names[idx] if idx >= 0 else "Unadmitted" for idx in assigned_slice_idx
        ]
        admitted = (assigned_slice_idx >= 0).astype(int)

        revenue = float(np.sum(admitted * prices))
        cost = float(np.sum(p_sol * costs.reshape(1, -1)))
        profit = revenue - cost
        weighted_satisfaction = float(np.sum(weights * satisfaction))
        objective_value = weighted_satisfaction + profit

        users["Assigned_Slice"] = assigned_slices
        users["Admitted"] = admitted
        users["Allocated_Rate_Mbps"] = np.round(allocated_rate, 4)
        users["Satisfaction"] = np.round(satisfaction, 4)
        users["Satisfied_Flag"] = satisfied_flag
        users["Tier_Target_Rate_Mbps"] = targets
        users["Tier_Weight"] = weights
        users["Tier_Price"] = prices

        for k, name in enumerate(self.slice_names):
            users[f"PRBs_{name}"] = np.round(p_sol[:, k], 4)

        slice_usage = []
        for k, name in enumerate(self.slice_names):
            used_prbs = float(np.sum(p_sol[:, k]))
            slice_usage.append(
                {
                    "Slice": name,
                    "Used_PRBs": round(used_prbs, 4),
                    "Capacity_PRBs": capacities[k],
                    "Utilization": round(used_prbs / capacities[k], 4) if capacities[k] else 0.0,
                }
            )

        return OptimzerResult(
            status=status,
            success=success,
            message=message,
            objective_value=objective_value,
            weighted_satisfaction=weighted_satisfaction,
            revenue=revenue,
            cost=cost,
            profit=profit,
            user_allocations=users,
            slice_usage=pd.DataFrame(slice_usage),
        )

    @abstractmethod
    def optimize(self, users_df: pd.DataFrame) -> OptimzerResult:
        raise NotImplementedError
