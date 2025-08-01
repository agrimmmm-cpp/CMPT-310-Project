import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """Custom Stock Trading Environment using Gymnasium"""
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance=10000, max_steps=200):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_steps = max_steps

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [Price, SMA_10, RSI_14, balance, shares_held]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # Initialize internal variables
        self.reset()

    def _get_price(self):
        return self.df.iloc[self.current_step]['Price']

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Price'],
            row['SMA_10'],
            row['RSI_14'],
            self.balance / self.initial_balance,
            self.shares_held / 100  # normalize
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance

        self.start_step = np.random.randint(0, len(self.df) - self.max_steps)
        self.current_step = self.start_step

        obs = self._get_observation()
        return obs, {}  # Gymnasium expects (obs, info)

    def step(self, action):
        price = self._get_price()
        terminated = False
        truncated = False

        # Execute action
        if action == 1:  # Buy
            if self.balance >= price:
                self.shares_held += 1
                self.balance -= price

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += price

        # Update portfolio value
        self.portfolio_value = self.balance + self.shares_held * price
        reward = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

        self.prev_portfolio_value = self.portfolio_value

        self.current_step += 1

        # End the episode if max steps reached
        if self.current_step >= self.start_step + self.max_steps:
            terminated = True

        obs = self._get_observation()

        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Shares: {self.shares_held} | Portfolio: {self.portfolio_value:.2f}")
