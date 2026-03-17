from dataclasses import dataclass

import pandas as pd


@dataclass
class OptimzerResult:
    status: int
    success: bool
    message: str
    objective_value: float
    weighted_satisfaction: float
    weighted_satisfaction_penalized: float
    revenue: float
    cost: float
    profit: float
    user_allocations: pd.DataFrame
    slice_usage: pd.DataFrame
    min_satisfied_violation: float = 0.0
