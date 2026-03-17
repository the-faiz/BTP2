from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from channel_model import Channel
from config_loader import load_config

CONFIG = load_config()
MAIN_CFG = CONFIG["main"]
USER_CFG = CONFIG["user_profile"]
TIERS_CFG = CONFIG["tiers"]


@dataclass
class User:
    user_id: int
    tier: str
    distance_m: float
    speed_kmh: float
    sinr_linear: float
    required_bw_mhz: float

    @staticmethod
    def generate_user_profile(
        n_users: int = MAIN_CFG["default_n_users"],
        cell_radius: float = USER_CFG["default_cell_radius"],
    ) -> pd.DataFrame:
        """
        Generate synthetic users and return them as a DataFrame.
        """
        tiers = TIERS_CFG
        mobility_speeds = USER_CFG["mobility"]["speeds_kmh"]
        mobility_probs = USER_CFG["mobility"]["probabilities"]
        tier_names = list(tiers.keys())
        tier_probs_cfg = USER_CFG.get("tier_probabilities", {})
        tier_probs = [float(tier_probs_cfg.get(name, 0.0)) for name in tier_names]
        total_prob = float(sum(tier_probs))
        if total_prob <= 0.0:
            tier_probs = None
        else:
            tier_probs = [p / total_prob for p in tier_probs]
        channel = Channel(cell_radius=cell_radius)

        theta = 2 * np.pi * np.random.rand(n_users)
        r = cell_radius * np.sqrt(np.random.rand(n_users))
        distances = r

        users = []
        for i in range(n_users):
            tier = np.random.choice(tier_names, p=tier_probs)
            target_rate = tiers[tier]["target_rate_mbps"]
            weight = tiers[tier]["weight"]
            speed = np.random.choice(mobility_speeds, p=mobility_probs)

            sinr_linear = channel.compute_sinr_linear(distance_m=distances[i])
            required_bw = target_rate / np.log2(1 + sinr_linear)

            users.append(
                User(
                    user_id=i + 1,
                    tier=tier,
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
                "distance_m": "Distance_m",
                "speed_kmh": "Speed_kmh",
                "sinr_linear": "SINR_linear",
                "required_bw_mhz": "Required_BW_MHz",
            }
        )
