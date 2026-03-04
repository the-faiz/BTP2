import numpy as np
import pandas as pd

from config_loader import load_config

CONFIG = load_config()
USER_CFG = CONFIG["user_profile"]
CHANNEL_CFG = CONFIG["channel"]


def generate_user_profile(
    n_users=USER_CFG["default_n_users"], cell_radius=USER_CFG["default_cell_radius"]
):
    """
    Generates a synthetic dataset for Multi-Slice Resource Allocation.
    """
    # 1. Initialize Constants
    P_tx_dbm = CHANNEL_CFG["p_tx_dbm"]  # Transmit power of Base Station (dBm)
    Noise_Floor_dbm = CHANNEL_CFG["thermal_noise_density_dbm_hz"] + 10 * np.log10(
        CHANNEL_CFG["noise_bandwidth_hz"]
    )  # Thermal noise for 180kHz PRB
    
    # Tier Data [cite: 12, 13, 14, 15]
    tiers = USER_CFG["tiers"]
    tier_names = list(tiers.keys())

    # 2. Spatial Distribution (Monte Carlo) [cite: 20, 47]
    # Uniformly distribute users in a circular cell
    theta = 2 * np.pi * np.random.rand(n_users)
    r = cell_radius * np.sqrt(np.random.rand(n_users))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    distances = r # Distance from Base Station at (0,0) 

    # 3. User Parameters
    data = []
    for i in range(n_users):
        # Assign Tier 
        tier = np.random.choice(tier_names)
        R_target = tiers[tier]['R_target']
        weight = tiers[tier]['weight']
        
        # Mobility (Speed v_i in km/h) 
        # Mix of stationary (0), pedestrian (3-5), and vehicular (30-60)
        v_i = np.random.choice(
            USER_CFG["mobility"]["speeds_kmh"], p=USER_CFG["mobility"]["probabilities"]
        )
        
        # 4. Signal Modeling & Interference [cite: 16, 21]
        # Simple Path Loss Model (Cost231 Hata or similar simplified)
        # PL = 128.1 + 37.6 * log10(d_km)
        dist_km = max(
            distances[i] / 1000, CHANNEL_CFG["min_distance_km"]
        )  # Avoid log(0)
        path_loss = CHANNEL_CFG["path_loss_offset"] + CHANNEL_CFG["path_loss_slope"] * np.log10(
            dist_km
        )
        
        P_rx_dbm = P_tx_dbm - path_loss
        P_rx_linear = 10**(P_rx_dbm / 10)
        
        # Modeling Interference (I) 
        # Increases slightly as user moves toward cell edge
        interference_dbm = CHANNEL_CFG["interference_base_dbm"] + (
            CHANNEL_CFG["interference_edge_gain_dbm"] * (distances[i] / cell_radius)
        )
        I_linear = 10**(interference_dbm / 10)
        N_linear = 10**(Noise_Floor_dbm / 10)
        
        # SINR calculation (gamma_i) [cite: 17]
        sinr_linear = P_rx_linear / (I_linear + N_linear)
        
        # 5. Required Bandwidth (B_i) via Shannon 
        # B_i = R_target / log2(1 + sinr)
        bandwidth_req = R_target / np.log2(1 + sinr_linear)
        
        data.append({
            'User_ID': i + 1,
            'Tier': tier,
            'Weight': weight,
            'Target_Rate_Mbps': R_target,
            'Distance_m': round(distances[i], 2),
            'Speed_kmh': v_i,
            'SINR_linear': round(sinr_linear, 4),
            'Required_BW_MHz': round(bandwidth_req, 4)
        })

    return pd.DataFrame(data)

# Generate and display
# df_users = generate_user_profile(n_users=20)
# print(df_users.head())
