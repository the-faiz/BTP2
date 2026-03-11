from dataclasses import dataclass

import pandas as pd


@dataclass
class OptimzerResult:
    status: int
    success: bool
    message: str
    objective_value: float
    weighted_satisfaction: float
    revenue: float
    cost: float
    profit: float
    user_allocations: pd.DataFrame
    slice_usage: pd.DataFrame
