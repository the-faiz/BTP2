#This file is the main orchestrator
from config_loader import load_config
from user_profile import User

CONFIG = load_config()

if __name__ == "__main__":
    # Generate user profiles
    df_users = User.generate_user_profile(n_users=CONFIG["main"]["default_n_users"])
    print(df_users.head())
