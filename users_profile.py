import numpy as np
import pandas as pd

def generate_user_profile(n_users=50, cell_radius=1000):
    """
    Generates a synthetic dataset for Multi-Slice Resource Allocation.
    """
    # 1. Initialize Constants
    P_tx_dbm = 46  # Transmit power of Base Station (dBm)
    Noise_Floor_dbm = -174 + 10 * np.log10(180e3) # Thermal noise for 180kHz PRB
    
    # Tier Data [cite: 12, 13, 14, 15]
    tiers = {
        'Gold':   {'R_target': 20, 'weight': 3},
        'Silver': {'R_target': 10, 'weight': 2},
        'Bronze': {'R_target': 5,  'weight': 1}
    }
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
        v_i = np.random.choice([0, 5, 40], p=[0.4, 0.4, 0.2]) 
        
        # 4. Signal Modeling & Interference [cite: 16, 21]
        # Simple Path Loss Model (Cost231 Hata or similar simplified)
        # PL = 128.1 + 37.6 * log10(d_km)
        dist_km = max(distances[i] / 1000, 0.01) # Avoid log(0)
        path_loss = 128.1 + 37.6 * np.log10(dist_km)
        
        P_rx_dbm = P_tx_dbm - path_loss
        P_rx_linear = 10**(P_rx_dbm / 10)
        
        # Modeling Interference (I) 
        # Increases slightly as user moves toward cell edge
        interference_dbm = -90 + (10 * (distances[i] / cell_radius)) 
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