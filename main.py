#This file is the main orchestrator
from user_profile import User

if __name__ == "__main__":
    # Generate user profiles
    df_users = User.generate_user_profile(n_users=20)
    print(df_users.head())
