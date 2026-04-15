import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lagrangian_config import LAGRANGIAN_HIGHWAY_CONFIG, LAGRANGIAN_HYPERPARAMS
from shared_core_config import DQN_HYPERPARAMS

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

os.makedirs("results/lagrangian", exist_ok=True)


# ── Classes ───────────────────────────────────────────────────────────────────

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
    def __init__(self, n_obs, n_actions, hidden_size=256, **kwargs):
        self.n_actions = n_actions
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_agent(get_action_fn, env, n_episodes=50, seed=100):
    rewards = []
    collisions = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        collided = False
        while not done:
            action = get_action_fn(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
            if term:
                collided = True
        rewards.append(total)
        if collided:
            collisions += 1
    return np.array(rewards), collisions


def smooth(x, w=50):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


EVAL_SEEDS = [100, 200, 300]
N_EVAL = 50


def full_eval(get_action_fn, env, name):
    all_rewards = []
    all_collisions = 0
    for seed in EVAL_SEEDS:
        r, c = eval_agent(get_action_fn, env, n_episodes=N_EVAL, seed=seed)
        all_rewards.append(r)
        all_collisions += c
    all_rewards = np.concatenate(all_rewards)
    coll_rate = all_collisions / (N_EVAL * len(EVAL_SEEDS))
    print(f"  {name:20s}  reward={all_rewards.mean():.3f}"
          f"  collision_rate={coll_rate:.1%}", flush=True)
    return float(all_rewards.mean()), float(all_rewards.std()), coll_rate


# ── Setup ─────────────────────────────────────────────────────────────────────

env = gym.make("highway-v0", render_mode=None)
env.unwrapped.configure(LAGRANGIAN_HIGHWAY_CONFIG)
obs, _ = env.reset(seed=GLOBAL_SEED)
n_obs = obs.size
n_actions = env.action_space.n

# Charge Lagrangian DQN
lagrangian = DQN(n_obs=n_obs, n_actions=n_actions,
                 hidden_size=LAGRANGIAN_HYPERPARAMS["hidden_size"])
lagrangian.load("results/lagrangian/lagrangian_final.pt")
print("Lagrangian DQN checkpoint loaded.", flush=True)

# Charge DQN classique
dqn = DQN(n_obs=n_obs, n_actions=n_actions,
          hidden_size=DQN_HYPERPARAMS["hidden_size"])
dqn.load("results/dqn/dqn_final.pt")
print("DQN classique checkpoint loaded.", flush=True)

# Charge Safety DQN
safety_dqn = DQN(n_obs=n_obs, n_actions=n_actions,
                 hidden_size=DQN_HYPERPARAMS["hidden_size"])
safety_dqn.load("results/safety_dqn/dqn_safety_final.pt")
print("Safety DQN checkpoint loaded.", flush=True)


# ── 1. Courbe d'entraînement ──────────────────────────────────────────────────

print("\nGenerating training curves...", flush=True)
lag_rewards = np.load("results/lagrangian/train_rewards.npy")
lag_lambdas = np.load("results/lagrangian/train_lambda.npy")
dqn_rewards = np.load("results/dqn/train_rewards.npy")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(smooth(dqn_rewards), label="DQN classique", color="steelblue")
axes[0].plot(smooth(lag_rewards), label="Lagrangian DQN", color="crimson")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward (smoothed w=50)")
axes[0].set_title("Courbes d'entrainement")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(lag_lambdas, color="crimson", alpha=0.7)
axes[1].axhline(y=lag_lambdas[-1], color="crimson", linestyle="--",
                label=f"lambda final = {lag_lambdas[-1]:.3f}")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Lambda")
axes[1].set_title("Evolution du multiplicateur lambda")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/lagrangian/training_curves.png", dpi=150)
plt.close()
print("  training curves saved.", flush=True)


# ── 2. Evaluation des 3 agents ────────────────────────────────────────────────

print("\nEvaluating all agents...", flush=True)

lag_mean, lag_std, lag_coll   = full_eval(lambda o: lagrangian.get_action(o), env, "Lagrangian DQN")
dqn_mean, dqn_std, dqn_coll   = full_eval(lambda o: dqn.get_action(o),        env, "DQN classique")
saf_mean, saf_std, saf_coll   = full_eval(lambda o: safety_dqn.get_action(o), env, "Safety DQN")


# ── 3. Graphe de comparaison ──────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Barres récompense
names  = ["DQN\nclassique", "Safety\nDQN", "Lagrangian\nDQN"]
means  = [dqn_mean, saf_mean, lag_mean]
stds   = [dqn_std,  saf_std,  lag_std]
colls  = [dqn_coll, saf_coll, lag_coll]
colors = ["steelblue", "green", "crimson"]

bars = axes[0].bar(names, means, yerr=stds, capsize=8,
                   color=colors, alpha=0.85)
axes[0].set_ylabel("Recompense moyenne (150 episodes)")
axes[0].set_title("Performance")
axes[0].grid(axis="y", alpha=0.3)
for bar, m in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=10)

# Barres collision
bars2 = axes[1].bar(names, colls, color=colors, alpha=0.85)
axes[1].axhline(y=LAGRANGIAN_HYPERPARAMS["collision_threshold"],
                color="black", linestyle="--",
                label=f"Contrainte ({LAGRANGIAN_HYPERPARAMS['collision_threshold']:.0%})")
axes[1].set_ylabel("Taux de collision")
axes[1].set_title("Securite")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
for bar, c in zip(bars2, colls):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{c:.1%}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("results/lagrangian/comparison.png", dpi=150)
plt.close()
print("  comparison plot saved.", flush=True)


# ── 4. Tableau recapitulatif ──────────────────────────────────────────────────

summary = pd.DataFrame([
    {"agent": "DQN classique",   "mean_reward": dqn_mean, "std": dqn_std, "collision_rate": dqn_coll},
    {"agent": "Safety DQN",      "mean_reward": saf_mean, "std": saf_std, "collision_rate": saf_coll},
    {"agent": "Lagrangian DQN",  "mean_reward": lag_mean, "std": lag_std, "collision_rate": lag_coll},
])
summary.to_csv("results/lagrangian/comparison.csv", index=False)

print("\n=== RESUME FINAL ===")
print(summary.to_string(index=False))
print(f"\nContrainte cible : collision_rate < {LAGRANGIAN_HYPERPARAMS['collision_threshold']:.0%}")
print(f"Lambda final     : {lag_lambdas[-1]:.4f}")
print("\nALL EVALUATION DONE.", flush=True)
