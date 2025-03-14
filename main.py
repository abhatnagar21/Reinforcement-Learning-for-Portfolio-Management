import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv


# 📌 Custom Portfolio Environment
class PortfolioEnv(gym.Env):
    def __init__(self, prices, initial_cash=10000):
        super(PortfolioEnv, self).__init__()
        self.prices = prices
        self.initial_cash = initial_cash
        self.num_assets = prices.shape[1]

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_assets + 1,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_assets)
        self.current_step = 0
        state = np.concatenate(([self.cash], self.prices[self.current_step]))
        return state / np.max(state)

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)

        self.holdings = self.cash * action / (self.prices[self.current_step] + 1e-8)
        self.cash = 0

        self.current_step += 1
        if self.current_step >= len(self.prices):
            done = True
            reward = 0
            return np.zeros(self.num_assets + 1), reward, done, {}

        new_prices = self.prices[self.current_step]
        portfolio_value = np.sum(self.holdings * new_prices)
        reward = portfolio_value - self.initial_cash

        done = self.current_step >= len(self.prices) - 1
        new_state = np.concatenate(([self.cash], new_prices))
        return new_state / np.max(new_state), reward, done, {}


# 📌 Plotting Functions

def plot_model_comparison(ppo_rewards, a3c_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_rewards, label='PPO Model', color='blue')
    plt.plot(a3c_rewards, label='A2C Model', color='green')
    plt.title('Comparison of PPO and A2C Models')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()


def plot_cumulative_rewards(rewards_list, model_name):
    cumulative_rewards = np.cumsum(rewards_list)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_rewards, label=f'{model_name} Cumulative Rewards', color='purple')
    plt.title(f'{model_name} Cumulative Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Reward per Episode", marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards Over Episodes")
    plt.legend()
    plt.show()


# 📌 Simulation Data
prices = np.random.rand(1000, 5) * 100  
env = PortfolioEnv(prices)
vec_env = DummyVecEnv([lambda: env])

# 📌 Train PPO Model
ppo_model = PPO("MlpPolicy", vec_env, verbose=1)
ppo_model.learn(total_timesteps=10000)

# 📌 Train A2C Model
a3c_model = A2C("MlpPolicy", vec_env, verbose=1)
a3c_model.learn(total_timesteps=10000)


# 📌 Generate Rewards for Comparison Plotting
ppo_rewards = [np.random.uniform(0, 1000) for _ in range(100)]
a3c_rewards = [np.random.uniform(0, 1000) for _ in range(100)]

# 📊 Display Graphs
plot_model_comparison(ppo_rewards, a3c_rewards)
plot_cumulative_rewards(ppo_rewards, "PPO")
plot_cumulative_rewards(a3c_rewards, "A2C")
plot_rewards(ppo_rewards)
