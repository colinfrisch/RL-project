import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import highway_env  # noqa: F401

from shared_core_config import HIGHWAY_CONFIG, DQN_HYPERPARAMS, SB3_HYPERPARAMS

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

os.makedirs('results/dqn', exist_ok=True)
os.makedirs('results/sb3', exist_ok=True)

CKPT_DQN   = 'results/dqn/dqn_final.pt'
CKPT_DQN_R = 'results/dqn/train_rewards.npy'
CKPT_SB3   = 'results/sb3/sb3_dqn'
CKPT_SB3_R = 'results/sb3/train_rewards.npy'


# ── Components ────────────────────────────────────────────────────────────────

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
        return random.choices(self.memory, k=batch_size)

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
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_steps': self.n_steps,
            'n_eps': self.n_eps,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location='cpu')
        self.q_net.load_state_dict(ck['q_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.n_steps = ck['n_steps']
        self.n_eps   = ck['n_eps']
        self.epsilon = ck['epsilon']


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_agent(get_action_fn, env, n_episodes=10, seed=100):
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            action = get_action_fn(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
    return np.array(rewards)


def train_dqn(agent, env, n_episodes, eval_freq=200, train_seed=0):
    ep_rewards = []
    t0 = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=train_seed + ep)
        ep_reward, done = 0.0, False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.update(obs, action, reward, done, next_obs)
            obs = next_obs
            ep_reward += reward
        agent.n_eps += 1
        agent.decrease_epsilon()
        ep_rewards.append(ep_reward)

        if (ep + 1) % eval_freq == 0:
            greedy_fn = lambda o: agent.get_action(o, epsilon=0.0)
            mean_r = eval_agent(greedy_fn, env, n_episodes=10).mean()
            elapsed = time.time() - t0
            print(
                f'[DQN {ep+1:4d}/{n_episodes}]'
                f'  train100={np.mean(ep_rewards[-100:]):.3f}'
                f'  eval={mean_r:.3f}'
                f'  eps={agent.epsilon:.3f}'
                f'  elapsed={elapsed:.0f}s',
                flush=True,
            )
    return ep_rewards


# ── Train Our DQN ─────────────────────────────────────────────────────────────

print('=' * 60)
print('PHASE 1 -- Training our DQN')
print('=' * 60, flush=True)

env = gym.make('highway-v0', render_mode=None)
env.unwrapped.configure(HIGHWAY_CONFIG)
obs, _ = env.reset(seed=GLOBAL_SEED)
N_OBS = obs.size
N_ACTIONS = env.action_space.n

if os.path.exists(CKPT_DQN):
    agent = DQN(n_obs=N_OBS, n_actions=N_ACTIONS, **DQN_HYPERPARAMS)
    agent.load(CKPT_DQN)
    print(f'Checkpoint found -- skipping DQN training.')
else:
    agent = DQN(n_obs=N_OBS, n_actions=N_ACTIONS, **DQN_HYPERPARAMS)
    rewards = train_dqn(agent, env, DQN_HYPERPARAMS['n_episodes'], train_seed=GLOBAL_SEED)
    agent.save(CKPT_DQN)
    np.save(CKPT_DQN_R, np.array(rewards))
    print(f'DQN saved to {CKPT_DQN}', flush=True)


# ── Train SB3 DQN ─────────────────────────────────────────────────────────────

print()
print('=' * 60)
print('PHASE 2 -- Training SB3 DQN')
print('=' * 60, flush=True)

from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self._cur = 0.0
        self._t0 = time.time()
        self._last_print = 0

    def _on_step(self):
        self._cur += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.ep_rewards.append(self._cur)
            self._cur = 0.0
            n = len(self.ep_rewards)
            if n - self._last_print >= 200:
                elapsed = time.time() - self._t0
                mean100 = np.mean(self.ep_rewards[-100:])
                print(
                    f'[SB3 ep {n:4d}]'
                    f'  train100={mean100:.3f}'
                    f'  elapsed={elapsed:.0f}s',
                    flush=True,
                )
                self._last_print = n
        return True


if os.path.exists(CKPT_SB3 + '.zip'):
    print('SB3 checkpoint found -- skipping SB3 training.')
else:
    sb3_env = gym.make('highway-v0', render_mode=None)
    sb3_env.unwrapped.configure(HIGHWAY_CONFIG)
    sb3_env.reset(seed=GLOBAL_SEED)
    sb3_env = Monitor(sb3_env)

    cb = RewardCallback()
    sb3_agent = SB3DQN(
        'MlpPolicy', sb3_env,
        gamma=SB3_HYPERPARAMS['gamma'],
        learning_rate=SB3_HYPERPARAMS['learning_rate'],
        batch_size=SB3_HYPERPARAMS['batch_size'],
        buffer_size=SB3_HYPERPARAMS['buffer_size'],
        learning_starts=SB3_HYPERPARAMS['learning_starts'],
        target_update_interval=SB3_HYPERPARAMS['target_update_interval'],
        train_freq=SB3_HYPERPARAMS['train_freq'],
        exploration_fraction=SB3_HYPERPARAMS['exploration_fraction'],
        exploration_initial_eps=SB3_HYPERPARAMS['exploration_initial_eps'],
        exploration_final_eps=SB3_HYPERPARAMS['exploration_final_eps'],
        policy_kwargs=SB3_HYPERPARAMS['policy_kwargs'],
        seed=GLOBAL_SEED,
        verbose=0,
    )
    sb3_agent.learn(total_timesteps=SB3_HYPERPARAMS['total_timesteps'], callback=cb)
    sb3_agent.save(CKPT_SB3)
    np.save(CKPT_SB3_R, np.array(cb.ep_rewards))
    print(f'SB3 saved to {CKPT_SB3}  ({len(cb.ep_rewards)} episodes)', flush=True)

print()
print('ALL TRAINING DONE', flush=True)
