import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import highway_env  # noqa: F401

from lagrangian_config import LAGRANGIAN_HIGHWAY_CONFIG, LAGRANGIAN_HYPERPARAMS

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

os.makedirs("results/lagrangian", exist_ok=True)

CKPT_DQN = "results/lagrangian/lagrangian_final.pt"
CKPT_REW = "results/lagrangian/train_rewards.npy"
CKPT_LAM = "results/lagrangian/train_lambda.npy"


# ── Composants ────────────────────────────────────────────────────────────────

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


class LagrangianDQN:
    def __init__(self, n_obs, n_actions, gamma, batch_size, buffer_capacity,
                 update_target_every, epsilon_start, decrease_epsilon_factor,
                 epsilon_min, learning_rate, hidden_size=256,
                 collision_threshold=0.05, lambda_lr=0.01,
                 lambda_init=0.0, lambda_max=10.0, **kwargs):

        self.n_obs = n_obs
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min

        # Lagrangian
        self.collision_threshold = collision_threshold
        self.lambda_lr = lambda_lr
        self.lambda_val = lambda_init
        self.lambda_max = lambda_max

        self.buffer = ReplayBuffer(buffer_capacity)
        self.q_net = Net(n_obs, hidden_size, n_actions).to(DEVICE)
        self.target_net = Net(n_obs, hidden_size, n_actions).to(DEVICE)
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
            s = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return self.q_net(s).argmax().item()

    def update_lambda(self, episode_collision_rate):
        """
        Mise à jour du multiplicateur de Lagrange après chaque épisode.
        Si trop de collisions → λ augmente → agent plus prudent
        Si pas assez de collisions → λ diminue → agent plus agressif
        """
        self.lambda_val += self.lambda_lr * (episode_collision_rate - self.collision_threshold)
        self.lambda_val = float(np.clip(self.lambda_val, 0.0, self.lambda_max))

    def update(self, state, action, reward, cost, done, next_state):
        """
        cost = 1 si collision ce step, 0 sinon
        reward_effective = reward - lambda * cost
        """
        effective_reward = reward - self.lambda_val * cost

        self.buffer.push(
            torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([effective_reward], dtype=torch.float32),
            torch.tensor([int(done)], dtype=torch.int64),
            torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        s_b, a_b, r_b, d_b, ns_b = (torch.cat(x) for x in zip(*transitions))
        s_b  = s_b.to(DEVICE)
        a_b  = a_b.to(DEVICE)
        r_b  = r_b.to(DEVICE)
        d_b  = d_b.to(DEVICE)
        ns_b = ns_b.to(DEVICE)

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
            "lambda_val": self.lambda_val,
        }, path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_agent(get_action_fn, env, n_episodes=10, seed=100):
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
    return np.array(rewards), collisions / n_episodes


def train_lagrangian(agent, env, n_episodes, eval_freq=200, train_seed=0):
    ep_rewards = []
    ep_lambdas = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=train_seed + ep)
        ep_reward, done = 0.0, False
        ep_collisions = 0
        ep_steps = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # coût = 1 si collision ce step
            cost = 1.0 if term else 0.0

            agent.update(obs, action, reward, cost, done, next_obs)
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            if term:
                ep_collisions += 1

        # Mise à jour de lambda après chaque épisode
        ep_collision_rate = ep_collisions / max(ep_steps, 1)
        agent.update_lambda(ep_collision_rate)

        agent.n_eps += 1
        agent.decrease_epsilon()
        ep_rewards.append(ep_reward)
        ep_lambdas.append(agent.lambda_val)

        if ep < 10:
            print(
                f"[LAG ep {ep+1:4d}] reward={ep_reward:.3f}"
                f"  lambda={agent.lambda_val:.3f}"
                f"  eps={agent.epsilon:.3f}",
                flush=True,
            )
        elif (ep + 1) % eval_freq == 0:
            greedy_fn = lambda o: agent.get_action(o, epsilon=0.0)
            eval_rewards, coll_rate = eval_agent(greedy_fn, env, n_episodes=10)
            elapsed = time.time() - t0
            print(
                f"[LAG {ep+1:4d}/{n_episodes}]"
                f"  train100={np.mean(ep_rewards[-100:]):.3f}"
                f"  eval={eval_rewards.mean():.3f}"
                f"  coll={coll_rate:.1%}"
                f"  lambda={agent.lambda_val:.3f}"
                f"  eps={agent.epsilon:.3f}"
                f"  elapsed={elapsed:.0f}s",
                flush=True,
            )

    return ep_rewards, ep_lambdas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(LAGRANGIAN_HIGHWAY_CONFIG)
    obs, _ = env.reset(seed=GLOBAL_SEED)
    n_obs = obs.size
    n_actions = env.action_space.n

    agent = LagrangianDQN(n_obs=n_obs, n_actions=n_actions, **LAGRANGIAN_HYPERPARAMS)
    rewards, lambdas = train_lagrangian(
        agent, env, LAGRANGIAN_HYPERPARAMS["n_episodes"], train_seed=GLOBAL_SEED
    )
    agent.save(CKPT_DQN)
    np.save(CKPT_REW, np.array(rewards))
    np.save(CKPT_LAM, np.array(lambdas))
    print(f"Saved Lagrangian DQN to {CKPT_DQN}", flush=True)
    print(f"Final lambda = {agent.lambda_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
