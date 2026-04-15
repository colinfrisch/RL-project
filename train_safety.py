import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import highway_env 

from safety_config import (
    SAFETY_HIGHWAY_CONFIG,
    SAFETY_REWARD_CONFIG,
    SAFETY_DQN_HYPERPARAMS,
)
from safety_reward_wrapper import SafetyRewardWrapper

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

os.makedirs("results/safety_dqn", exist_ok=True)

CKPT_DQN = "results/safety_dqn/dqn_safety_final.pt"
CKPT_REW = "results/safety_dqn/train_rewards.npy"
CKPT_PEN = "results/safety_dqn/train_safety_penalties.npy"

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.memory), size=batch_size, replace=False)
        return [self.memory[i] for i in idx]

    def __len__(self):
        return len(self.memory)

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
    def __init__(self, n_obs, n_actions, gamma, batch_size, buffer_capacity,
                 update_target_every, epsilon_start, decrease_epsilon_factor,
                 epsilon_min, learning_rate, hidden_size=256, **kwargs):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min

        self.buffer = ReplayBuffer(buffer_capacity)
        self.q_net = Net(n_obs, hidden_size, n_actions)
        self.target_net = Net(n_obs, hidden_size, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.epsilon = epsilon_start
        self.n_steps = 0
        self.n_eps = 0

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).argmax().item()

    def update(self, state, action, reward, done, next_state):
        self.buffer.push(
            torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([int(done)], dtype=torch.int64),
            torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        s_b, a_b, r_b, d_b, ns_b = (torch.cat(x) for x in zip(*transitions))

        q_vals = self.q_net(s_b).gather(1, a_b)

        with torch.no_grad():
            next_q = (1 - d_b) * self.target_net(ns_b).max(1)[0]
            targets = r_b + self.gamma * next_q

        loss = self.loss_fn(q_vals, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_steps += 1
        if self.n_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (
            (self.epsilon_start - self.epsilon_min)
            * np.exp(-self.n_eps / self.decrease_epsilon_factor)
        )

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "n_steps": self.n_steps,
            "n_eps": self.n_eps,
            "epsilon": self.epsilon,
        }, path)

def make_env():
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(SAFETY_HIGHWAY_CONFIG)
    env = SafetyRewardWrapper(env, **SAFETY_REWARD_CONFIG)
    return env

def eval_agent(get_action_fn, env, n_episodes=10, seed=100):
    rewards = []
    collisions = 0
    penalties = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total, total_penalty, done = 0.0, 0.0, False
        while not done:
            action = get_action_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            total_penalty += info.get("safety_penalty", 0.0)
            done = term or trunc
            if term:
                collisions += 1
        rewards.append(total)
        penalties.append(total_penalty)
    return np.array(rewards), collisions / n_episodes, np.array(penalties)

def train_dqn(agent, env, n_episodes, eval_freq=200, train_seed=0):
    ep_rewards = []
    ep_penalties = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=train_seed + ep)
        ep_reward, ep_penalty, done = 0.0, 0.0, False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            agent.update(obs, action, reward, done, next_obs)

            ep_reward += reward
            ep_penalty += info.get("safety_penalty", 0.0)
            obs = next_obs

        agent.n_eps += 1
        agent.decrease_epsilon()
        ep_rewards.append(ep_reward)
        ep_penalties.append(ep_penalty)

        if ep < 10:
            print(
                f"[SAFETY DQN ep {ep+1:4d}] reward={ep_reward:.3f} "
                f"safety_pen={ep_penalty:.3f} eps={agent.epsilon:.3f}",
                flush=True,
            )
        elif (ep + 1) % eval_freq == 0:
            greedy_fn = lambda o: agent.get_action(o, epsilon=0.0)
            eval_rewards, collision_rate, eval_penalties = eval_agent(
                greedy_fn, env, n_episodes=10
            )
            print(
                f"[SAFETY DQN {ep+1:4d}/{n_episodes}] "
                f"train100={np.mean(ep_rewards[-100:]):.3f} "
                f"eval={eval_rewards.mean():.3f} "
                f"coll_rate={collision_rate:.2%} "
                f"eval_pen={eval_penalties.mean():.3f} "
                f"eps={agent.epsilon:.3f} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    return ep_rewards, ep_penalties

def main():
    env = make_env()
    obs, _ = env.reset(seed=GLOBAL_SEED)
    n_obs = obs.size
    n_actions = env.action_space.n

    agent = DQN(n_obs=n_obs, n_actions=n_actions, **SAFETY_DQN_HYPERPARAMS)
    rewards, penalties = train_dqn(
        agent, env, SAFETY_DQN_HYPERPARAMS["n_episodes"], train_seed=GLOBAL_SEED
    )
    agent.save(CKPT_DQN)
    np.save(CKPT_REW, np.array(rewards))
    np.save(CKPT_PEN, np.array(penalties))

    print(f"Saved safety DQN to {CKPT_DQN}", flush=True)

if __name__ == "__main__":
    main()