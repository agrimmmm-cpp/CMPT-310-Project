# main.py

import pandas as pd
from environment.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent
from stable_baselines3.common.monitor import Monitor


# 🔹 Load cleaned full dataset
df = pd.read_csv("data/processed/AAPL_cleaned.csv")

# 🔹 Split into training and testing sets (no shuffle!)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]

# After you create your custom environment
env = TradingEnv(train_df)
env = Monitor(env, filename="./logs/monitor.csv")  # <-- This enables scalar logging

# 🔹 Train agent
agent = DQNAgent(env)
agent.train(timesteps=10000)
agent.save("outputs/models/dqn_trading_model")
