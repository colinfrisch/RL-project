import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_core_config import HIGHWAY_CONFIG, DQN_HYPERPARAMS
from safety_config import (
    SAFETY_HIGHWAY_CONFIG,
    SAFETY_REWARD_CONFIG,
    SAFETY_EVAL_DENSE_CONFIG,
    SAFETY_EVAL_HARD_CONFIG,
)
from safety_reward_wrapper import SafetyRewardWrapper

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

os.makedirs("results/safety_eval", exist_ok=True)

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class DQN:
    def __init__(self, n_obs, n_actions, hidden_size=256):
        self.q_net = Net(n_obs, hidden_size, n_actions)
        self.target_net = Net(n_obs, hidden_size, n_actions)

    def get_action(self, state):
        with torch.no_grad():
            s = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).argmax().item()

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.q_net.load_state_dict(ck["q_net"])
        self.target_net.load_state_dict(ck["target_net"])

def make_env(env_config, use_safety_wrapper=False):
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(env_config)
    if use_safety_wrapper:
        env = SafetyRewardWrapper(env, **SAFETY_REWARD_CONFIG)
    return env

def evaluate(agent_fn, env, n_episodes=50, seed=100):
    rewards = []
    collisions = []
    lengths = []
    penalties = []
    lane_changes = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total_reward = 0.0
        total_penalty = 0.0
        total_lane_changes = 0
        steps = 0
        collided = False

        while not done:
            action = agent_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            total_reward += r
            total_penalty += info.get("safety_penalty", 0.0)
            total_lane_changes += int(info.get("lane_changed", False))
            steps += 1
            done = term or trunc
            if term:
                collided = True

        rewards.append(total_reward)
        collisions.append(int(collided))
        lengths.append(steps)
        penalties.append(total_penalty)
        lane_changes.append(total_lane_changes)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "collision_rate": float(np.mean(collisions)),
        "mean_length": float(np.mean(lengths)),
        "mean_penalty": float(np.mean(penalties)),
        "mean_lane_changes": float(np.mean(lane_changes)),
    }

def main():
    # baseline DQN core model
    base_env = make_env(HIGHWAY_CONFIG, use_safety_wrapper=True)
    obs, _ = base_env.reset(seed=GLOBAL_SEED)
    n_obs = obs.size
    n_actions = base_env.action_space.n

    core_dqn = DQN(n_obs, n_actions, hidden_size=DQN_HYPERPARAMS["hidden_size"])
    core_dqn.load("results/dqn/dqn_final.pt")

    safety_dqn = DQN(n_obs, n_actions, hidden_size=DQN_HYPERPARAMS["hidden_size"])
    safety_dqn.load("results/safety_dqn/dqn_safety_final.pt")

    agents = {
        "core_dqn": lambda obs: core_dqn.get_action(obs),
        "safety_dqn": lambda obs: safety_dqn.get_action(obs),
    }

    configs = {
        "standard": SAFETY_HIGHWAY_CONFIG,
        "dense": SAFETY_EVAL_DENSE_CONFIG,
        "hard": SAFETY_EVAL_HARD_CONFIG,
    }

    rows = []

    for config_name, config in configs.items():
        for agent_name, agent_fn in agents.items():
            env = make_env(config, use_safety_wrapper=True)
            metrics = evaluate(agent_fn, env, n_episodes=50, seed=100)
            row = {"config": config_name, "agent": agent_name}
            row.update(metrics)
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/safety_eval/safety_results.csv", index=False)

    # simple plot: collision rate
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot(index="config", columns="agent", values="collision_rate")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety evaluation")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/safety_eval/collision_rate_comparison.png", dpi=150)
    plt.close()

    # mean reward
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot(index="config", columns="agent", values="mean_reward")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean shaped reward")
    ax.set_title("Reward comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/safety_eval/reward_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()