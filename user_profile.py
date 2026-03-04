from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from channel_model import Channel


@dataclass
class User:
    user_id: int
    tier: str
    weight: int
    target_rate_mbps: float
    distance_m: float
    speed_kmh: float
    sinr_linear: float
    required_bw_mhz: float

    @staticmethod
    def generate_user_profile(n_users: int = 50, cell_radius: float = 1000.0) -> pd.DataFrame:
        """
        Generate synthetic users and return them as a DataFrame.
        """
        tiers = {
            "Gold": {"R_target": 20, "weight": 3},
            "Silver": {"R_target": 10, "weight": 2},
            "Bronze": {"R_target": 5, "weight": 1},
        }
        tier_names = list(tiers.keys())
        channel = Channel(cell_radius=cell_radius)

        theta = 2 * np.pi * np.random.rand(n_users)
        r = cell_radius * np.sqrt(np.random.rand(n_users))
        distances = r

        users = []
        for i in range(n_users):
            tier = np.random.choice(tier_names)
            target_rate = tiers[tier]["R_target"]
            weight = tiers[tier]["weight"]
            speed = np.random.choice([0, 5, 40], p=[0.4, 0.4, 0.2])

            sinr_linear = channel.compute_sinr_linear(distance_m=distances[i])
            required_bw = target_rate / np.log2(1 + sinr_linear)

            users.append(
                User(
                    user_id=i + 1,
                    tier=tier,
                    weight=weight,
                    target_rate_mbps=float(target_rate),
                    distance_m=round(float(distances[i]), 2),
                    speed_kmh=float(speed),
                    sinr_linear=round(float(sinr_linear), 4),
                    required_bw_mhz=round(float(required_bw), 4),
                )
            )

        return pd.DataFrame(asdict(user) for user in users).rename(
            columns={
                "user_id": "User_ID",
                "tier": "Tier",
                "weight": "Weight",
                "target_rate_mbps": "Target_Rate_Mbps",
                "distance_m": "Distance_m",
                "speed_kmh": "Speed_kmh",
                "sinr_linear": "SINR_linear",
                "required_bw_mhz": "Required_BW_MHz",
            }
        )

