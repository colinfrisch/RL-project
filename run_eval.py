import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from shared_core_config import HIGHWAY_CONFIG, DQN_HYPERPARAMS, SB3_HYPERPARAMS

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# ── DQN classes ───────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.capacity: self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity
    def sample(self, k): return random.choices(self.memory, k=k)
    def __len__(self): return len(self.memory)

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
    def forward(self, x): return self.net(x)

class DQN:
    def __init__(self, n_obs, n_actions, hidden_size=256, **kw):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.q_net = Net(n_obs, hidden_size, n_actions)
        self.target_net = Net(n_obs, hidden_size, n_actions)
        self.epsilon = 0.0
        self.n_steps = 0
        self.n_eps = 0
    def get_action(self, state, epsilon=None):
        if epsilon is None: epsilon = self.epsilon
        if np.random.rand() < epsilon: return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).argmax().item()
    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.q_net.load_state_dict(ck['q_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.n_steps = ck['n_steps']
        self.n_eps = ck['n_eps']
        self.epsilon = ck['epsilon']

# ── Setup ─────────────────────────────────────────────────────────────────────

env = gym.make('highway-v0', render_mode=None)
env.unwrapped.configure(HIGHWAY_CONFIG)
obs, _ = env.reset(seed=GLOBAL_SEED)
N_OBS = obs.size
N_ACTIONS = env.action_space.n
ACTION_NAMES = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}

agent = DQN(n_obs=N_OBS, n_actions=N_ACTIONS, hidden_size=DQN_HYPERPARAMS['hidden_size'])
agent.load('results/dqn/dqn_final.pt')
print('DQN checkpoint loaded.', flush=True)

from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.common.monitor import Monitor
sb3_env = gym.make('highway-v0', render_mode=None)
sb3_env.unwrapped.configure(HIGHWAY_CONFIG)
sb3_env.reset(seed=GLOBAL_SEED)
sb3_env = Monitor(sb3_env)
sb3_agent = SB3DQN.load('results/sb3/sb3_dqn', env=sb3_env)
print('SB3 checkpoint loaded.', flush=True)

# ── Eval helpers ──────────────────────────────────────────────────────────────

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

def smooth(x, w=50):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

# ── 1. Training curves ───────────────────────────────────────────────────────

print('Generating training curves...', flush=True)
dqn_arr = np.load('results/dqn/train_rewards.npy')
sb3_arr = np.load('results/sb3/train_rewards.npy')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(smooth(dqn_arr), label='Our DQN (smoothed, w=50)', color='steelblue')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Total reward')
axes[0].set_title('DQN -- Training Reward'); axes[0].legend(); axes[0].grid(alpha=0.3)
eps_curve = [DQN_HYPERPARAMS['epsilon_min'] + (1.0 - DQN_HYPERPARAMS['epsilon_min'])
             * np.exp(-ep / DQN_HYPERPARAMS['decrease_epsilon_factor'])
             for ep in range(len(dqn_arr))]
axes[1].plot(eps_curve, color='orange', label='epsilon')
axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Epsilon')
axes[1].set_title('Epsilon Decay Schedule'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/dqn/training_curve.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(smooth(dqn_arr), label='Our DQN', color='steelblue')
ax.plot(smooth(sb3_arr), label='SB3 DQN', color='seagreen')
ax.set_xlabel('Episode'); ax.set_ylabel('Total reward (smoothed, w=50)')
ax.set_title('Training Curves -- Our DQN vs SB3 DQN')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/training_comparison.png', dpi=150); plt.close()
print('  training curves saved.', flush=True)

# ── 2. Evaluation (3 seeds x 50 runs) ────────────────────────────────────────

EVAL_SEEDS = [100, 200, 300]
N_EVAL = 50

eval_env = gym.make('highway-v0', render_mode=None)
eval_env.unwrapped.configure(HIGHWAY_CONFIG)
eval_env.reset()

def full_eval(get_action_fn, env, name, seeds=EVAL_SEEDS, n_eps=N_EVAL):
    per_seed = {}
    for s in seeds:
        t0 = time.time()
        per_seed[s] = eval_agent(get_action_fn, env, n_episodes=n_eps, seed=s)
        print(f'  {name} seed={s}  mean={per_seed[s].mean():.3f}  ({time.time()-t0:.0f}s)', flush=True)
    all_r = np.concatenate(list(per_seed.values()))
    return per_seed, float(all_r.mean()), float(all_r.std())

random_fn = lambda obs: np.random.randint(N_ACTIONS)
dqn_fn    = lambda obs: agent.get_action(obs, epsilon=0.0)
sb3_fn    = lambda obs: int(sb3_agent.predict(obs, deterministic=True)[0])

print('Evaluating Random...', flush=True)
rand_per, rand_mean, rand_std = full_eval(random_fn, eval_env, 'Random')
print('Evaluating Our DQN...', flush=True)
dqn_per, dqn_mean, dqn_std = full_eval(dqn_fn, eval_env, 'DQN')
print('Evaluating SB3 DQN...', flush=True)
sb3_per, sb3_mean, sb3_std = full_eval(sb3_fn, eval_env, 'SB3')

def fmt_row(per_seed, mean, std):
    row = {f'seed {s}': f'{v.mean():.3f} +/- {v.std():.3f}' for s, v in per_seed.items()}
    row['Overall (150 runs)'] = f'{mean:.3f} +/- {std:.3f}'
    return row

table = pd.DataFrame({
    'Random':  fmt_row(rand_per, rand_mean, rand_std),
    'Our DQN': fmt_row(dqn_per,  dqn_mean,  dqn_std),
    'SB3 DQN': fmt_row(sb3_per,  sb3_mean,  sb3_std),
}).T
print('\n' + table.to_string() + '\n', flush=True)
table.to_csv('results/evaluation_table.csv')

fig, ax = plt.subplots(figsize=(7, 5))
names  = ['Random', 'Our DQN', 'SB3 DQN']
means  = [rand_mean, dqn_mean, sb3_mean]
stds   = [rand_std,  dqn_std,  sb3_std]
colors = ['#888888', 'steelblue', 'seagreen']
bars = ax.bar(names, means, yerr=stds, capsize=8, color=colors, alpha=0.85)
ax.set_ylabel('Mean reward (150 evaluation runs)'); ax.set_title('Evaluation Comparison')
ax.grid(axis='y', alpha=0.3)
for bar, m in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{m:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout(); plt.savefig('results/evaluation_bar_chart.png', dpi=150); plt.close()
print('  evaluation chart saved.', flush=True)

# ── 3. Behavior analysis ─────────────────────────────────────────────────────

print('Running behavior analysis...', flush=True)

def run_behavior(get_action_fn, env, n_episodes=30, seed=200):
    action_counts = np.zeros(N_ACTIONS, dtype=int)
    ep_lengths, ep_rewards, collisions = [], [], 0
    speeds = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total, steps, done = 0.0, 0, False
        while not done:
            action = get_action_fn(obs)
            action_counts[action] += 1
            speeds.append(obs[0, 3])
            obs, r, term, trunc, _ = env.step(action)
            total += r; steps += 1; done = term or trunc
            if term: collisions += 1
        ep_lengths.append(steps); ep_rewards.append(total)
    return action_counts, ep_lengths, ep_rewards, collisions, speeds

ac, ep_len, ep_rew, collisions, speeds = run_behavior(dqn_fn, eval_env)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
labels = [ACTION_NAMES[i] for i in range(N_ACTIONS)]
axes[0].bar(labels, ac/ac.sum(), color='steelblue', alpha=0.8)
axes[0].set_title('Action Distribution'); axes[0].set_ylabel('Fraction of steps')
axes[0].tick_params(axis='x', rotation=20); axes[0].grid(axis='y', alpha=0.3)
axes[1].hist(ep_len, bins=10, color='steelblue', alpha=0.8, edgecolor='white')
axes[1].set_title('Episode Length Distribution'); axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Count'); axes[1].grid(alpha=0.3)
axes[2].plot(ep_rew, marker='o', color='steelblue', alpha=0.7, markersize=4)
axes[2].axhline(np.mean(ep_rew), color='red', linestyle='--', label=f'mean={np.mean(ep_rew):.3f}')
axes[2].set_title('Reward per Episode'); axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Total reward'); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/behavior_analysis.png', dpi=150); plt.close()

max_dur = HIGHWAY_CONFIG['duration']
print(f'  Collision rate  : {collisions/30:.1%}  ({collisions}/30 episodes)')
print(f'  Mean ep. length : {np.mean(ep_len):.1f} steps  (max={max_dur})')
print(f'  Mean speed (vx) : {np.mean(speeds):.3f}  (normalized)', flush=True)

# ── 4. Failure mode ──────────────────────────────────────────────────────────

print('Searching for failure mode...', flush=True)

def find_failure(get_action_fn, env, max_attempts=200, seed=500):
    for i in range(max_attempts):
        obs_hist, act_hist = [], []
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action = get_action_fn(obs)
            obs_hist.append(obs.copy()); act_hist.append(action)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            if term: return obs_hist, act_hist, seed + i
    return None, None, None

obs_hist, act_hist, fail_seed = find_failure(dqn_fn, eval_env)

if act_hist is not None:
    print(f'  Collision at seed={fail_seed}, step={len(act_hist)}')
    print(f'  Last 10 actions: {[ACTION_NAMES[a] for a in act_hist[-10:]]}', flush=True)
    ego_vx = [o[0, 3] for o in obs_hist]
    n1_vx  = [o[1, 3] for o in obs_hist]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(ego_vx, label='Ego vx', color='steelblue')
    axes[0].plot(n1_vx, label='Nearest vehicle vx', color='orange', linestyle='--')
    axes[0].axvline(len(ego_vx)-1, color='red', linestyle=':', label='Collision')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Normalized speed')
    axes[0].set_title('Failure Episode -- Speed Profile'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].step(range(len(act_hist)), act_hist, color='steelblue', where='post')
    axes[1].axvline(len(act_hist)-1, color='red', linestyle=':', label='Collision')
    axes[1].set_yticks(range(N_ACTIONS))
    axes[1].set_yticklabels([ACTION_NAMES[i] for i in range(N_ACTIONS)])
    axes[1].set_xlabel('Step'); axes[1].set_title('Failure Episode -- Action Sequence')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('results/failure_mode.png', dpi=150); plt.close()
else:
    print('  No collision found.', flush=True)

print('\nALL EVALUATION DONE.', flush=True)
