import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from environment.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

# 🔹 Output directory for plots
output_dir = "outputs/evaluation"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Load cleaned data
df = pd.read_csv("data/processed/AAPL_cleaned.csv")

# 🔹 Use the last 20% for testing
train_size = int(len(df) * 0.8)
test_df = df.iloc[train_size:]

# 🔹 Create the environment with test data
env = TradingEnv(test_df)

# 🔹 Load the trained agent (without .zip!)
agent = DQNAgent(env, model_path="outputs/models/dqn_trading_model")

# 🔹 Run evaluation
obs, _ = env.reset()
done = False

portfolio_values = [env.initial_balance]
actions = []

while not done:
    action, _ = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    portfolio_values.append(env.portfolio_value)
    actions.append(action)

# 🔹 Evaluation Metrics
initial_value = env.initial_balance
final_value = env.portfolio_value
cumulative_return = (final_value - initial_value) / initial_value * 100

buy_count = actions.count(1)
sell_count = actions.count(2)
hold_count = actions.count(0)

# 🔹 Print Results
print("\n✅ Evaluation Results:")
print(f"Initial Portfolio Value: ${initial_value:.2f}")
print(f"Final Portfolio Value:   ${final_value:.2f}")
print(f"Cumulative Return:       {cumulative_return:.2f}%")
print(f"Actions taken: Buy={buy_count}, Sell={sell_count}, Hold={hold_count}")

# 🔹 Save Portfolio Plot
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label="Portfolio Value", color='blue')
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Step")
plt.ylabel("Value ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "portfolio_value.png"))
plt.close()

# 🔹 Save Action Distribution
plt.figure(figsize=(6, 4))
labels = ["Hold", "Buy", "Sell"]
counts = [hold_count, buy_count, sell_count]
plt.bar(labels, counts, color=["gray", "green", "red"])
plt.title("Action Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "action_distribution.png"))
plt.close()

print(f"\n📁 Plots saved to: {output_dir}")
