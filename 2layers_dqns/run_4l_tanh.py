import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import pandas as pd

FEATURE_NAMES  = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
FEATURE_LABELS = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Ang. Vel"]

# ── Network (4 layers, tanh) ──────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, env, hidden_dim=64):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        n_actions = env.action_space.n
        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return self.layer4(x)

    @property
    def w1(self): return self.layer1.weight
    @property
    def w2(self): return self.layer2.weight
    @property
    def w3(self): return self.layer3.weight
    @property
    def w4(self): return self.layer4.weight


# ── Feature contribution (gradient-based for tanh) ────────────────────────────
def get_feature_contributions(q_network, state):
    """Returns q_values (n_actions,), contributions (n_actions, n_feats), dominant_feat (n_actions,)."""
    x = torch.FloatTensor(state).requires_grad_(True)
    q = q_network(x)
    n_actions = q.shape[0]
    n_features = x.shape[0]
    jac = torch.zeros(n_actions, n_features)
    for a in range(n_actions):
        if x.grad is not None:
            x.grad.zero_()
        q[a].backward(retain_graph=(a < n_actions - 1))
        jac[a] = x.grad.detach().clone()
    contributions = (jac * x.detach().unsqueeze(0)).numpy()
    q_values = q.detach().numpy()
    dominant_feat = np.argmax(np.abs(contributions), axis=1)
    return q_values, contributions, dominant_feat


# ── Hyperparameters ───────────────────────────────────────────────────────────
LR            = 1e-3
GAMMA         = 0.99
BATCH_SIZE    = 64
BUFFER_SIZE   = 10000
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE = 500
N_EPISODES    = 20000

# ── Setup ─────────────────────────────────────────────────────────────────────
env            = gym.make("CartPole-v1")
q_network      = QNetwork(env)
target_network = QNetwork(env)
target_network.load_state_dict(q_network.state_dict())
optimizer      = optim.Adam(q_network.parameters(), lr=LR)
memory         = deque(maxlen=BUFFER_SIZE)
epsilon        = EPSILON_START

step_log    = []   # one dict per env step
episode_log = []   # one dict per episode
global_step = 0

# ── Training loop ─────────────────────────────────────────────────────────────
for episode in range(N_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        q_values, contributions, dominant_feat = get_feature_contributions(q_network, state)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_values))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        total_reward += reward

        # Training step
        loss_val = None
        if len(memory) > BATCH_SIZE:
            s_b, a_b, r_b, ns_b, d_b = zip(*random.sample(memory, BATCH_SIZE))
            s_b  = torch.FloatTensor(np.array(s_b))
            a_b  = torch.LongTensor(a_b).unsqueeze(1)
            r_b  = torch.FloatTensor(r_b)
            ns_b = torch.FloatTensor(np.array(ns_b))
            d_b  = torch.FloatTensor(d_b)

            current_q  = q_network(s_b).gather(1, a_b).squeeze()
            with torch.no_grad():
                target_q = r_b + (1 - d_b) * GAMMA * target_network(ns_b).max(1)[0]

            loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        # Record step
        row = {
            "episode":      episode,
            "global_step":  global_step,
            "action":       action,
            "epsilon":      epsilon,
            "loss":         loss_val,
            "q_left":       float(q_values[0]),
            "q_right":      float(q_values[1]),
            "q_chosen":     float(q_values[action]),
            "dominant_feat": FEATURE_NAMES[dominant_feat[action]],
        }
        for fi, fname in enumerate(FEATURE_NAMES):
            row[f"contrib_left_{fname}"]   = float(contributions[0, fi])
            row[f"contrib_right_{fname}"]  = float(contributions[1, fi])
            row[f"contrib_chosen_{fname}"] = float(contributions[action, fi])

        step_log.append(row)
        state = next_state
        global_step += 1

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    if episode % TARGET_UPDATE == 0:
        target_network.load_state_dict(q_network.state_dict())
        print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | w1 norm: {q_network.w1.norm().item():.2f}")

    episode_log.append({"episode": episode, "reward": total_reward})

env.close()

# Build DataFrames
df_steps = pd.DataFrame(step_log)
df_ep    = pd.DataFrame(episode_log)

# Episode-level aggregates (mean per episode)
df_ep_agg = df_steps.groupby("episode").agg(
    mean_loss=("loss",     "mean"),
    mean_q=("q_chosen",   "mean"),
    **{f"mean_contrib_{f}": (f"contrib_chosen_{f}", lambda x: x.abs().mean()) for f in FEATURE_NAMES}
).reset_index()
df_ep_agg = df_ep_agg.merge(df_ep, on="episode")

print(f"\nCollected {len(df_steps):,} steps across {N_EPISODES} episodes.")