# agents/dqn_agent.py

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

class DQNAgent:
    def __init__(self, env, model_path=None):
        self.env = DummyVecEnv([lambda: env])  # required for SB3
        self.model_path = model_path

        if model_path:
            self.model = DQN.load(model_path, env=self.env)
        else:
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=0.0001,
                buffer_size=5000,
                learning_starts=1000,
                batch_size=32,
                tau=0.1,
                gamma=0.95,
                train_freq=1,
                target_update_interval=100,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                tensorboard_log="./logs/"  # <-- Enables TB logging
            )

    def train(self, timesteps=10000):
        # EvalCallback for logging and saving best model
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path="./outputs/best_model/",
            log_path="./logs/",
            eval_freq=500,
            deterministic=True,
            render=False
        )

        self.model.learn(
            total_timesteps=timesteps,
            callback=eval_callback  # <-- Injects logging
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = DQN.load(path, env=self.env)

    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)
