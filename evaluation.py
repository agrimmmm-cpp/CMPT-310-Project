import pandas as pd
import matplotlib.pyplot as plt
from environment.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

# 🔹 Load cleaned dataset
df = pd.read_csv("data/processed/AAPL_cleaned.csv")

# 🔹 Use the last 20% for testing
train_size = int(len(df) * 0.8)
test_df = df.iloc[train_size:].reset_index(drop=True)

# 🔹 Create test environment
env = TradingEnv(test_df)

# 🔹 Load trained agent from .zip file
agent = DQNAgent(env, model_path="outputs/models/dqn_trading_model")

# 🔹 Reset environment for evaluation
obs, _ = env.reset()
done = False

# 🔹 Track metrics
portfolio_values = []
actions = []
rewards = []

while not done:
    # Predict next action using trained model
    action, _ = agent.predict(obs)  # no 'deterministic' arg needed in your wrapper
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    portfolio_values.append(info["portfolio_value"])
    actions.append(action)
    rewards.append(reward)

# 🔹 Evaluation summary
initial_value = portfolio_values[0]
final_value = portfolio_values[-1]
cumulative_return = (final_value - initial_value) / initial_value * 100

buy_count = actions.count(1)
sell_count = actions.count(2)
hold_count = actions.count(0)

print(f"\n✅ Evaluation Results:")
print(f"Initial Portfolio Value: ${initial_value:.2f}")
print(f"Final Portfolio Value:   ${final_value:.2f}")
print(f"Cumulative Return:       {cumulative_return:.2f}%")
print(f"Actions taken: Buy={buy_count}, Sell={sell_count}, Hold={hold_count}\n")

# 🔹 Plot portfolio value over time
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title('📈 Portfolio Value Over Time')
plt.xlabel('Step')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔹 Plot actions taken over time
plt.figure(figsize=(10, 3))
plt.plot(actions, label='Action', linestyle='--', marker='o')
plt.yticks([0, 1, 2], ['Hold', 'Buy', 'Sell'])
plt.title('🎯 Agent Actions Over Time')
plt.xlabel('Step')
plt.ylabel('Action')
plt.grid(True)
plt.tight_layout()
plt.show()
