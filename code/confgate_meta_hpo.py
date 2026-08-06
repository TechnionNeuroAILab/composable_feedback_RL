"""
Multi-task meta-RL hyperparameter controller for DQN.

Outer loop: sample a task, keep a shared GRU controller.
Inner loop: fresh DQN per task; controller samples major training knobs
            (all categorical): lr_mult, eps_end, eps_start,
            epsilon_decay_episodes, train_freq, target_freq — applied to Adam /
            ε-schedules (never via loss scaling). Controller trained with
            REINFORCE + value baseline on return-improvement reward.

Tasks (default): CartPole-v1, Acrobot-v1, MountainCar-v0

Usage:
    python code/confgate_meta_hpo.py \\
        --tasks CartPole-v1,Acrobot-v1,MountainCar-v0 \\
        --meta-iters 50 --inner-steps 50000 --seeds 1

    python code/confgate_meta_hpo.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "paper" / "_tmp_b_feedb_cg" / "ckpt_meta_hpo"
FIG_DIR = ROOT / "paper" / "figures" / "conf_meta_hpo"

# ---------------------------------------------------------------------------
# Hyper-parameters (CleanRL-style defaults)
# ---------------------------------------------------------------------------
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
BUFFER_SIZE = 10_000
BATCH_SIZE = 128
START_E = 1.0
END_E = 0.05
EPSILON_DECAY_EPISODES = 500  # shorter for inner budgets
LEARNING_STARTS = 1_000
DEFAULT_TRAIN_FREQ = 10
DEFAULT_TARGET_FREQ = 500

OBS_TO_120 = 120
HIDDEN_84 = 84
CTRL_HIDDEN = 32
TASK_EMB_DIM = 4

DEFAULT_TASKS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
DEFAULT_SEEDS = [1]
DEFAULT_META_ITERS = 50
DEFAULT_INNER_STEPS = 50_000
META_INTERVAL = 64  # env steps between controller actions / rewards
CTRL_LR = 1e-3
ENTROPY_COEF = 0.05  # higher: six categorical heads need more exploration
VALUE_COEF = 0.5
STEP_COST = 0.01
RETURN_EMA_ALPHA = 0.1
META_REWARD_CLIP = 5.0  # clip Δreturn reward for PG stability across tasks
HEAD_BIAS_PREF = 0.5  # mild prior toward CleanRL-ish defaults (weights not zeroed)

TRAIN_FREQ_CHOICES = [1, 2, 4, 8, 10]
TARGET_FREQ_CHOICES = [1, 50, 100, 250, 500]

# 10-way grids for formerly continuous / fixed ε schedule knobs
LR_MULT_CHOICES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
EPS_END_CHOICES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.2]
EPS_START_CHOICES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPS_DECAY_EPISODES_CHOICES = [50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3500]

LR_MULT_LO, LR_MULT_HI = float(LR_MULT_CHOICES[0]), float(LR_MULT_CHOICES[-1])
EPS_END_LO, EPS_END_HI = float(EPS_END_CHOICES[0]), float(EPS_END_CHOICES[-1])
EPS_START_LO, EPS_START_HI = float(EPS_START_CHOICES[0]), float(EPS_START_CHOICES[-1])
EPS_DECAY_LO = float(EPS_DECAY_EPISODES_CHOICES[0])
EPS_DECAY_HI = float(EPS_DECAY_EPISODES_CHOICES[-1])

COLOR_BASE = "#2ca02c"
COLOR_META = "#ff7f0e"

KNOB_PLOT_KEYS = [
    "lr_mult",
    "eps_end",
    "eps_start",
    "epsilon_decay_episodes",
    "train_freq",
    "target_freq",
]

# Meta-state layout (no task emb):
#   loss, grad, return, eps, progress,
#   lr_n, eps_end_n, eps_start_n, eps_decay_n, train_freq_n, target_freq_n
BASE_STATE_DIM = 11


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _linear_schedule(start: float, end: float, duration_ep: int, n_ep_done: int) -> float:
    frac = min(1.0, n_ep_done / max(1, duration_ep))
    return start + frac * (end - start)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid") if w >= 2 else arr


def _parse_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_seeds(s: str) -> List[int]:
    return sorted({int(p.strip()) for p in s.split(",") if p.strip()})


def _nearest_choice_idx(choices: Sequence[float], value: float) -> int:
    return int(np.argmin([abs(float(c) - float(value)) for c in choices]))


def _choice_norm(choices: Sequence[float], value: float) -> float:
    idx = _nearest_choice_idx(choices, value)
    return idx / max(1, len(choices) - 1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
ReplaySamples = namedtuple(
    "ReplaySamples",
    ["observations", "actions", "next_observations", "dones", "rewards"],
)


class ReplayBuffer:
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action: int, reward: float, done: float) -> None:
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int) -> ReplaySamples:
        upper = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        return ReplaySamples(
            observations=torch.tensor(self.observations[idx], device=self.device),
            actions=torch.tensor(self.actions[idx], device=self.device).unsqueeze(1),
            next_observations=torch.tensor(self.next_observations[idx], device=self.device),
            dones=torch.tensor(self.dones[idx], device=self.device).unsqueeze(1),
            rewards=torch.tensor(self.rewards[idx], device=self.device).unsqueeze(1),
        )


# ---------------------------------------------------------------------------
# DQN agent (inner learner)
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, OBS_TO_120),
            nn.ReLU(),
            nn.Linear(OBS_TO_120, HIDDEN_84),
            nn.ReLU(),
            nn.Linear(HIDDEN_84, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNAgent:
    """Fresh Q-network + target + Adam; accepts live hparams each step."""

    def __init__(self, obs_dim: int, n_actions: int, device: torch.device):
        self.device = device
        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.t_net = QNetwork(obs_dim, n_actions).to(device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.last_loss = 0.0
        self.last_grad_norm = 0.0

    def act(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.q_net.net[-1].out_features)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(x).argmax(dim=1).item())

    def set_lr(self, lr_mult: float) -> None:
        self.opt.param_groups[0]["lr"] = LEARNING_RATE * float(lr_mult)

    def sync_target(self) -> None:
        self.t_net.load_state_dict(self.q_net.state_dict())

    def td_update(self, batch: ReplaySamples) -> float:
        with torch.no_grad():
            next_q = self.t_net(batch.next_observations).max(dim=1).values
            y = batch.rewards.flatten() + GAMMA * (1.0 - batch.dones.flatten()) * next_q
        q_sa = self.q_net(batch.observations).gather(1, batch.actions).squeeze(1)
        loss = F.mse_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.opt.step()
        self.last_loss = float(loss.item())
        self.last_grad_norm = float(gn.item() if hasattr(gn, "item") else gn)
        return self.last_loss


# ---------------------------------------------------------------------------
# Meta-HPO controller (outer shared policy)
# ---------------------------------------------------------------------------
class MetaAction:
    __slots__ = (
        "lr_mult",
        "eps_end",
        "eps_start",
        "epsilon_decay_episodes",
        "train_freq",
        "target_freq",
        "log_prob",
        "entropy",
        "value",
    )

    def __init__(
        self,
        lr_mult: float,
        eps_end: float,
        eps_start: float,
        epsilon_decay_episodes: int,
        train_freq: int,
        target_freq: int,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
    ):
        self.lr_mult = lr_mult
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.train_freq = train_freq
        self.target_freq = target_freq
        self.log_prob = log_prob
        self.entropy = entropy
        self.value = value


class MetaHPOController(nn.Module):
    """
    GRU policy with all-categorical knob heads:
      lr_mult, eps_end, eps_start, epsilon_decay_episodes, train_freq, target_freq.
    Weights keep default Linear init (not zeroed); mild bias prior toward baselines.
    """

    def __init__(self, n_tasks: int, state_dim: int = BASE_STATE_DIM):
        super().__init__()
        self.n_tasks = n_tasks
        self.state_dim = state_dim
        self.task_emb = nn.Embedding(n_tasks, TASK_EMB_DIM)
        in_dim = state_dim + TASK_EMB_DIM
        self.gru = nn.GRUCell(in_dim, CTRL_HIDDEN)
        self.trunk = nn.Sequential(nn.Linear(CTRL_HIDDEN, CTRL_HIDDEN), nn.ReLU())
        self.lr_head = nn.Linear(CTRL_HIDDEN, len(LR_MULT_CHOICES))
        self.eps_end_head = nn.Linear(CTRL_HIDDEN, len(EPS_END_CHOICES))
        self.eps_start_head = nn.Linear(CTRL_HIDDEN, len(EPS_START_CHOICES))
        self.eps_decay_head = nn.Linear(CTRL_HIDDEN, len(EPS_DECAY_EPISODES_CHOICES))
        self.tf_head = nn.Linear(CTRL_HIDDEN, len(TRAIN_FREQ_CHOICES))
        self.tgt_head = nn.Linear(CTRL_HIDDEN, len(TARGET_FREQ_CHOICES))
        self.value_head = nn.Linear(CTRL_HIDDEN, 1)
        self._init_heads()

    @staticmethod
    def _soft_prefer(head: nn.Linear, preferred_idx: int) -> None:
        """Mild logit bias toward a default; leave weight init intact (no pinning)."""
        with torch.no_grad():
            head.bias.zero_()
            if 0 <= preferred_idx < head.bias.numel():
                head.bias[preferred_idx] = HEAD_BIAS_PREF

    def _init_heads(self) -> None:
        self._soft_prefer(self.lr_head, LR_MULT_CHOICES.index(1.0))
        self._soft_prefer(self.eps_end_head, EPS_END_CHOICES.index(0.05))
        self._soft_prefer(self.eps_start_head, EPS_START_CHOICES.index(1.0))
        self._soft_prefer(
            self.eps_decay_head, EPS_DECAY_EPISODES_CHOICES.index(500)
        )
        self._soft_prefer(self.tf_head, TRAIN_FREQ_CHOICES.index(DEFAULT_TRAIN_FREQ))
        self._soft_prefer(
            self.tgt_head, TARGET_FREQ_CHOICES.index(DEFAULT_TARGET_FREQ)
        )

    def zero_hidden(
        self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(batch, CTRL_HIDDEN, device=device, dtype=dtype)

    def act(
        self,
        state: torch.Tensor,  # (B, state_dim)
        task_id: torch.Tensor,  # (B,) long
        h_prev: torch.Tensor,  # (B, H)
        deterministic: bool = False,
    ) -> Tuple[MetaAction, torch.Tensor]:
        emb = self.task_emb(task_id)
        x = torch.cat([state, emb], dim=-1)
        h = self.gru(x, h_prev)
        feat = self.trunk(h)

        dists = (
            Categorical(logits=self.lr_head(feat)),
            Categorical(logits=self.eps_end_head(feat)),
            Categorical(logits=self.eps_start_head(feat)),
            Categorical(logits=self.eps_decay_head(feat)),
            Categorical(logits=self.tf_head(feat)),
            Categorical(logits=self.tgt_head(feat)),
        )
        if deterministic:
            idxs = [d.probs.argmax(dim=-1) for d in dists]
        else:
            idxs = [d.sample() for d in dists]

        log_prob = sum(d.log_prob(i) for d, i in zip(dists, idxs))
        entropy = sum(d.entropy() for d in dists)
        value = self.value_head(feat).squeeze(-1)

        i0 = int(idxs[0][0].item())
        i1 = int(idxs[1][0].item())
        i2 = int(idxs[2][0].item())
        i3 = int(idxs[3][0].item())
        i4 = int(idxs[4][0].item())
        i5 = int(idxs[5][0].item())

        action = MetaAction(
            lr_mult=float(LR_MULT_CHOICES[i0]),
            eps_end=float(EPS_END_CHOICES[i1]),
            eps_start=float(EPS_START_CHOICES[i2]),
            epsilon_decay_episodes=int(EPS_DECAY_EPISODES_CHOICES[i3]),
            train_freq=int(TRAIN_FREQ_CHOICES[i4]),
            target_freq=int(TARGET_FREQ_CHOICES[i5]),
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )
        return action, h.detach()


# ---------------------------------------------------------------------------
# Meta-state builder
# ---------------------------------------------------------------------------
def build_meta_state(
    loss_ema: float,
    grad_ema: float,
    return_ema: float,
    eps: float,
    progress: float,
    lr_mult: float,
    eps_end: float,
    eps_start: float,
    epsilon_decay_episodes: int,
    train_freq: int,
    target_freq: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize knobs to roughly [0, 1] scales."""
    vec = np.array(
        [
            np.tanh(loss_ema),
            np.tanh(grad_ema / 10.0),
            np.tanh(return_ema / 100.0),
            float(eps),
            float(np.clip(progress, 0.0, 1.0)),
            _choice_norm(LR_MULT_CHOICES, lr_mult),
            _choice_norm(EPS_END_CHOICES, eps_end),
            _choice_norm(EPS_START_CHOICES, eps_start),
            _choice_norm(EPS_DECAY_EPISODES_CHOICES, float(epsilon_decay_episodes)),
            _choice_norm(TRAIN_FREQ_CHOICES, float(train_freq)),
            _choice_norm(TARGET_FREQ_CHOICES, float(target_freq)),
        ],
        dtype=np.float32,
    )
    return torch.tensor(vec, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Controller PG update helper
# ---------------------------------------------------------------------------
def _controller_step(
    log_prob: torch.Tensor,
    value: torch.Tensor,
    entropy: torch.Tensor,
    reward: float,
    controller: MetaHPOController,
    opt_ctrl: optim.Optimizer,
    update_controller: bool,
) -> float:
    """One-step REINFORCE + value baseline (avoids stale graphs across opt steps)."""
    if not update_controller:
        return 0.0
    lp = log_prob.view(-1)[0]
    v = value.view(-1)[0]
    ent = entropy.view(-1)[0]
    # Soft-bound reward so MountainCar/Acrobot Δreturn spikes don't explode PG
    reward = float(np.clip(reward, -META_REWARD_CLIP, META_REWARD_CLIP))
    ret = torch.tensor(reward, dtype=torch.float32, device=lp.device)
    advantage = ret - v.detach()
    pg_loss = -(advantage * lp)
    v_loss = F.mse_loss(v, ret)
    ctrl_loss = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * ent
    opt_ctrl.zero_grad()
    ctrl_loss.backward()
    nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)
    opt_ctrl.step()
    return float(ctrl_loss.item())


# ---------------------------------------------------------------------------
# Inner-task training
# ---------------------------------------------------------------------------
def run_inner_task(
    env_id: str,
    task_idx: int,
    controller: MetaHPOController,
    opt_ctrl: optim.Optimizer,
    device: torch.device,
    inner_steps: int,
    seed: int,
    use_controller: bool = True,
    update_controller: bool = True,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
    learning_starts: int = LEARNING_STARTS,
    fixed_knobs: Optional[Dict] = None,
    deterministic_ctrl: bool = False,
    total_episodes: Optional[int] = None,
) -> Dict:
    """
    Train a fresh DQN on one task. If use_controller, sample knobs via the
    shared meta-controller and (optionally) update it with REINFORCE + baseline.
    """
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim, n_actions, device)
    rb = ReplayBuffer(BUFFER_SIZE, env.observation_space.shape, device)

    episode_returns: List[float] = []
    knob_history: List[Dict] = []

    h = controller.zero_hidden(1, device)
    task_id_t = torch.tensor([task_idx], device=device, dtype=torch.long)

    if fixed_knobs is not None:
        lr_mult = float(fixed_knobs.get("lr_mult", 1.0))
        eps_end = float(fixed_knobs.get("eps_end", END_E))
        eps_start = float(fixed_knobs.get("eps_start", START_E))
        epsilon_decay_episodes = int(
            fixed_knobs.get("epsilon_decay_episodes", epsilon_decay_episodes)
        )
        train_freq = int(fixed_knobs.get("train_freq", DEFAULT_TRAIN_FREQ))
        target_freq = int(fixed_knobs.get("target_freq", DEFAULT_TARGET_FREQ))
    else:
        lr_mult, eps_end = 1.0, END_E
        eps_start = START_E
        train_freq, target_freq = DEFAULT_TRAIN_FREQ, DEFAULT_TARGET_FREQ

    agent.set_lr(lr_mult)

    loss_ema = 0.0
    grad_ema = 0.0
    return_ema = 0.0
    prev_return_ema = 0.0
    return_ema_ready = False  # True after first finished episode (avoids 0→-200 spike)
    steps_since_meta = 0
    n_meta_actions = 0
    ctrl_loss_sum = 0.0
    ctrl_loss_count = 0

    pending_log_prob: Optional[torch.Tensor] = None
    pending_value: Optional[torch.Tensor] = None
    pending_entropy: Optional[torch.Tensor] = None

    step_cap = max(inner_steps, 15_000_000) if total_episodes is not None else inner_steps
    obs, _ = env.reset(seed=seed)
    t = 0
    while t < step_cap and (
        total_episodes is None or len(episode_returns) < total_episodes
    ):
        n_ep = len(episode_returns)
        eps = _linear_schedule(eps_start, eps_end, epsilon_decay_episodes, n_ep)
        action = agent.act(obs, eps)

        next_obs, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        real_nxt = next_obs.copy()
        if truncated and "final_observation" in infos:
            real_nxt = infos["final_observation"]
        rb.add(obs, real_nxt, action, float(reward), float(done))

        if "episode" in infos:
            ret = float(np.asarray(infos["episode"]["r"]).item())
            episode_returns.append(ret)
            if not return_ema_ready:
                return_ema = ret
                prev_return_ema = ret
                return_ema_ready = True
            else:
                return_ema = (1 - RETURN_EMA_ALPHA) * return_ema + RETURN_EMA_ALPHA * ret

        obs = next_obs
        if done:
            obs, _ = env.reset()

        steps_since_meta += 1
        do_meta = use_controller and (steps_since_meta >= META_INTERVAL or t == 0)
        if do_meta:
            steps_since_meta = 0
            if pending_log_prob is not None and return_ema_ready:
                r = (return_ema - prev_return_ema) - STEP_COST
                loss_v = _controller_step(
                    pending_log_prob,
                    pending_value,  # type: ignore[arg-type]
                    pending_entropy,  # type: ignore[arg-type]
                    r,
                    controller,
                    opt_ctrl,
                    update_controller=update_controller,
                )
                if update_controller:
                    ctrl_loss_sum += loss_v
                    ctrl_loss_count += 1
                pending_log_prob = None
                pending_value = None
                pending_entropy = None
                prev_return_ema = return_ema
            elif pending_log_prob is not None and not return_ema_ready:
                # Drop unpaired action before any episode finished (no valid reward)
                pending_log_prob = None
                pending_value = None
                pending_entropy = None

            progress = (
                len(episode_returns) / max(1, total_episodes)
                if total_episodes is not None
                else t / max(1, inner_steps)
            )
            state = build_meta_state(
                loss_ema,
                grad_ema,
                return_ema,
                eps,
                progress,
                lr_mult,
                eps_end,
                eps_start,
                epsilon_decay_episodes,
                train_freq,
                target_freq,
                device,
            )
            # Avoid retaining graphs during eval / frozen controller
            if update_controller:
                meta_act, h = controller.act(
                    state, task_id_t, h, deterministic=deterministic_ctrl
                )
                pending_log_prob = meta_act.log_prob
                pending_value = meta_act.value
                pending_entropy = meta_act.entropy
            else:
                with torch.no_grad():
                    meta_act, h = controller.act(
                        state, task_id_t, h, deterministic=deterministic_ctrl
                    )
            lr_mult = meta_act.lr_mult
            eps_end = meta_act.eps_end
            eps_start = meta_act.eps_start
            epsilon_decay_episodes = meta_act.epsilon_decay_episodes
            train_freq = meta_act.train_freq
            target_freq = meta_act.target_freq
            agent.set_lr(lr_mult)
            n_meta_actions += 1

            if t % (META_INTERVAL * 10) == 0:
                knob_history.append(
                    {
                        "step": t,
                        "episode": len(episode_returns),
                        "lr_mult": lr_mult,
                        "eps_end": eps_end,
                        "eps_start": eps_start,
                        "epsilon_decay_episodes": epsilon_decay_episodes,
                        "train_freq": train_freq,
                        "target_freq": target_freq,
                        "return_ema": return_ema,
                    }
                )

        if t > learning_starts and t % max(1, train_freq) == 0 and len(rb) >= BATCH_SIZE:
            loss = agent.td_update(rb.sample(BATCH_SIZE))
            loss_ema = loss if loss_ema == 0.0 else 0.9 * loss_ema + 0.1 * loss
            grad_ema = (
                agent.last_grad_norm
                if grad_ema == 0.0
                else 0.9 * grad_ema + 0.1 * agent.last_grad_norm
            )

        if t % max(1, target_freq) == 0:
            agent.sync_target()

        t += 1

    if total_episodes is not None and len(episode_returns) < total_episodes:
        print(
            f"  WARNING {env_id}: hit step cap {step_cap:,} "
            f"with only {len(episode_returns)}/{total_episodes} episodes",
            flush=True,
        )

    env.close()

    if use_controller and pending_log_prob is not None and return_ema_ready:
        r = (return_ema - prev_return_ema) - STEP_COST
        loss_v = _controller_step(
            pending_log_prob,
            pending_value,  # type: ignore[arg-type]
            pending_entropy,  # type: ignore[arg-type]
            r,
            controller,
            opt_ctrl,
            update_controller=update_controller,
        )
        if update_controller:
            ctrl_loss_sum += loss_v
            ctrl_loss_count += 1

    last100 = (
        float(np.mean(episode_returns[-100:]))
        if len(episode_returns) >= 100
        else float(np.mean(episode_returns))
        if episode_returns
        else 0.0
    )
    return {
        "env_id": env_id,
        "task_idx": task_idx,
        "episode_returns": episode_returns,
        "knob_history": knob_history,
        "last100_mean": last100,
        "ctrl_loss": ctrl_loss_sum / max(1, ctrl_loss_count),
        "n_meta_steps": n_meta_actions,
        "use_controller": use_controller,
    }


# ---------------------------------------------------------------------------
# Outer meta-loop
# ---------------------------------------------------------------------------
def run_meta_train(
    tasks: Sequence[str],
    meta_iters: int,
    inner_steps: int,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    total_episodes: Optional[int] = None,
    tag: str = "",
    resume: bool = True,
    save_every: int = 1,
) -> Dict:
    """
    Train the shared meta-controller for `meta_iters` outer rounds.

    If resume=True and a checkpoint exists with fewer completed rounds, continue
    from len(history) up to meta_iters (extendable later by raising meta_iters).
    """
    _set_seed(seed)
    n_tasks = len(tasks)
    tag_sfx = f"_{tag}" if tag else ""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"meta_controller_seed{seed}{tag_sfx}.pt"

    controller = MetaHPOController(n_tasks=n_tasks).to(device)
    opt_ctrl = optim.Adam(controller.parameters(), lr=CTRL_LR)
    history: List[Dict] = []
    per_task_curves: Dict[str, List[List[float]]] = {t: [] for t in tasks}
    start_meta = 0

    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        if list(blob.get("tasks", [])) != list(tasks):
            raise ValueError(
                f"Checkpoint tasks {blob.get('tasks')} != requested {list(tasks)}"
            )
        controller.load_state_dict(blob["controller"])
        if "opt_ctrl" in blob:
            opt_ctrl.load_state_dict(blob["opt_ctrl"])
        history = list(blob.get("history", []))
        per_task_curves = blob.get("per_task_curves", per_task_curves)
        start_meta = len(history)
        print(
            f"Resuming seed={seed} tag={tag or '-'} from meta_iter "
            f"{start_meta}/{meta_iters} ({ckpt_path})",
            flush=True,
        )
        if start_meta >= meta_iters:
            print(
                f"Already at/above target meta_iters={meta_iters}; nothing to train.",
                flush=True,
            )
            return {
                "controller": controller,
                "history": history,
                "per_task_curves": per_task_curves,
                "ckpt_path": ckpt_path,
                "completed_meta_iters": start_meta,
            }

    print("=" * 60, flush=True)
    print("confgate_meta_hpo — multi-task meta-RL HPO", flush=True)
    print(f"  Tasks       : {list(tasks)}", flush=True)
    print(f"  Meta iters  : {start_meta} → {meta_iters}", flush=True)
    if total_episodes is not None:
        step_cap = max(inner_steps, int(total_episodes) * 2000)
        print(f"  Episodes    : {total_episodes:,} (step cap {step_cap:,})", flush=True)
    else:
        print(f"  Inner steps : {inner_steps:,}", flush=True)
    print(f"  Seed        : {seed}", flush=True)
    print(f"  Tag         : {tag or '-'}", flush=True)
    print(f"  Device      : {device}", flush=True)
    print("=" * 60, flush=True)

    def _save_ckpt(done_iters: int) -> None:
        torch.save(
            {
                "controller": controller.state_dict(),
                "opt_ctrl": opt_ctrl.state_dict(),
                "tasks": list(tasks),
                "seed": seed,
                "tag": tag,
                "meta_iters_target": meta_iters,
                "meta_iters_done": done_iters,
                "inner_steps": inner_steps,
                "total_episodes": total_episodes,
                "history": history,
                "per_task_curves": per_task_curves,
            },
            ckpt_path,
        )

    t0 = time.perf_counter()
    task_order = list(range(n_tasks))
    random.shuffle(task_order)
    for meta_i in range(start_meta, meta_iters):
        if meta_i % n_tasks == 0 and meta_i > 0:
            random.shuffle(task_order)
        task_idx = task_order[meta_i % n_tasks]
        env_id = tasks[task_idx]
        inner_seed = seed * 1000 + meta_i

        result = run_inner_task(
            env_id=env_id,
            task_idx=task_idx,
            controller=controller,
            opt_ctrl=opt_ctrl,
            device=device,
            inner_steps=inner_steps,
            seed=inner_seed,
            use_controller=True,
            total_episodes=total_episodes,
        )
        per_task_curves[env_id].append(result["episode_returns"])
        history.append(
            {
                "meta_iter": meta_i,
                "env_id": env_id,
                "last100_mean": result["last100_mean"],
                "ctrl_loss": result["ctrl_loss"],
                "n_episodes": len(result["episode_returns"]),
                "n_meta_steps": result["n_meta_steps"],
                "knob_history": result["knob_history"],
            }
        )
        print(
            f"  [meta {meta_i+1}/{meta_iters}] {env_id:16s}  "
            f"eps={len(result['episode_returns']):4d}  "
            f"last100={result['last100_mean']:7.1f}  "
            f"ctrl_loss={result['ctrl_loss']:.4f}",
            flush=True,
        )
        done = meta_i + 1
        if save_every > 0 and (done % save_every == 0 or done == meta_iters):
            _save_ckpt(done)
            print(f"  Checkpoint saved ({done}/{meta_iters}): {ckpt_path}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Meta-train done in {elapsed:.1f}s", flush=True)
    _save_ckpt(len(history))
    print(f"  Checkpoint: {ckpt_path}", flush=True)

    summary_path = checkpoint_dir / f"meta_train_summary_seed{seed}{tag_sfx}.json"
    summary = {
        "seed": seed,
        "tag": tag,
        "tasks": list(tasks),
        "meta_iters": meta_iters,
        "meta_iters_done": len(history),
        "inner_steps": inner_steps,
        "total_episodes": total_episodes,
        "history": [
            {k: v for k, v in h.items() if k != "knob_history"} for h in history
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "controller": controller,
        "history": history,
        "per_task_curves": per_task_curves,
        "ckpt_path": ckpt_path,
        "completed_meta_iters": len(history),
    }


# ---------------------------------------------------------------------------
# Eval: meta controller (frozen policy, still samples / deterministic) vs baseline
# ---------------------------------------------------------------------------
def run_eval_vs_baseline(
    tasks: Sequence[str],
    controller: Optional[MetaHPOController],
    inner_steps: int,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    deterministic_ctrl: bool = True,
    total_episodes: Optional[int] = None,
    tag: str = "",
) -> Dict[str, Dict[str, Dict]]:
    """
    For each task, run fixed-hparam baseline and (if controller) meta-controlled
    inner training. Meta eval does NOT update the controller.
    """
    results: Dict[str, Dict[str, Dict]] = {t: {} for t in tasks}
    fixed = {
        "lr_mult": 1.0,
        "eps_end": END_E,
        "eps_start": START_E,
        "epsilon_decay_episodes": EPSILON_DECAY_EPISODES,
        "train_freq": DEFAULT_TRAIN_FREQ,
        "target_freq": DEFAULT_TARGET_FREQ,
    }

    if controller is None:
        controller = MetaHPOController(n_tasks=len(tasks)).to(device)
    # lr=0 so accidental step is a no-op; update_controller=False skips backward
    frozen_opt = optim.Adam(controller.parameters(), lr=0.0)
    tag_sfx = f"_{tag}" if tag else ""
    # Raise step safety cap when stopping by episodes
    if total_episodes is not None:
        inner_steps = max(inner_steps, int(total_episodes) * 2000)

    for task_idx, env_id in enumerate(tasks):
        print(f"\n>>> EVAL baseline  {env_id}", flush=True)
        base = run_inner_task(
            env_id=env_id,
            task_idx=task_idx,
            controller=controller,
            opt_ctrl=frozen_opt,
            device=device,
            inner_steps=inner_steps,
            seed=seed + 17 + task_idx,
            use_controller=False,
            update_controller=False,
            fixed_knobs=fixed,
            total_episodes=total_episodes,
        )
        results[env_id]["baseline"] = base

        print(f">>> EVAL meta-ctrl {env_id}", flush=True)
        meta = run_inner_task(
            env_id=env_id,
            task_idx=task_idx,
            controller=controller,
            opt_ctrl=frozen_opt,
            device=device,
            inner_steps=inner_steps,
            seed=seed + 17 + task_idx,
            use_controller=True,
            update_controller=False,
            deterministic_ctrl=deterministic_ctrl,
            total_episodes=total_episodes,
        )
        results[env_id]["meta"] = meta

        print(
            f"  {env_id}: baseline last100={base['last100_mean']:.1f}"
            f"  meta last100={meta['last100_mean']:.1f}",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_path = checkpoint_dir / f"eval_vs_baseline_seed{seed}{tag_sfx}.pt"
    serializable = {
        env_id: {
            cond: {
                "episode_returns": res["episode_returns"],
                "last100_mean": res["last100_mean"],
                "knob_history": res.get("knob_history", []),
            }
            for cond, res in conds.items()
        }
        for env_id, conds in results.items()
    }
    torch.save(
        {
            "seed": seed,
            "tag": tag,
            "tasks": list(tasks),
            "total_episodes": total_episodes,
            "results": serializable,
        },
        eval_path,
    )
    print(f"  Eval checkpoint: {eval_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_eval_curves(
    results: Dict[str, Dict[str, Dict]],
    out_jpg: Path,
    title_suffix: str = "",
) -> None:
    tasks = list(results.keys())
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, env_id in zip(axes[0], tasks):
        for tag, color, label in (
            ("baseline", COLOR_BASE, "fixed hparams"),
            ("meta", COLOR_META, "meta-RL ctrl"),
        ):
            if tag not in results[env_id]:
                continue
            rets = np.asarray(results[env_id][tag]["episode_returns"], dtype=np.float64)
            if rets.size == 0:
                continue
            w = max(1, len(rets) // 50)
            sm = _smooth(rets, w)
            ep = np.arange(1, len(sm) + 1)
            last100 = results[env_id][tag]["last100_mean"]
            ax.plot(ep, sm, color=color, lw=2.0, label=f"{label} (last100={last100:.1f})")
        ax.set_title(env_id, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episodic return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Meta-HPO vs fixed hparams{title_suffix}", fontsize=12, y=1.02
    )
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_knob_trajectories(
    results: Dict[str, Dict[str, Dict]],
    out_jpg: Path,
) -> None:
    rows = []
    for env_id, tags in results.items():
        if "meta" not in tags:
            continue
        hist = tags["meta"].get("knob_history", [])
        if hist:
            rows.append((env_id, hist))
    if not rows:
        print("[knob plot] no knob_history — skipping.", flush=True)
        return

    keys = KNOB_PLOT_KEYS
    fig, axes = plt.subplots(
        len(keys), len(rows), figsize=(4.5 * len(rows), 2.0 * len(keys)), squeeze=False
    )
    for col, (env_id, hist) in enumerate(rows):
        steps = [h["step"] for h in hist]
        for row, key in enumerate(keys):
            ax = axes[row][col]
            if key not in hist[0]:
                ax.set_ylabel(key, fontsize=8)
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center")
                continue
            vals = [h[key] for h in hist]
            ax.plot(steps, vals, color=COLOR_META, lw=1.5)
            ax.set_ylabel(key, fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(env_id, fontsize=10)
            if row == len(keys) - 1:
                ax.set_xlabel("Inner step", fontsize=9)
    fig.suptitle("Sampled hyperparameter trajectories (meta-RL)", fontsize=12)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def _interp_hist(hist: List[Dict], x_key: str, y_key: str, grid: np.ndarray) -> np.ndarray:
    xs = np.array([float(h[x_key]) for h in hist])
    ys = np.array([float(h[y_key]) for h in hist])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux: List[float] = []
    uy: List[float] = []
    for x, y in zip(xs, ys):
        if ux and abs(x - ux[-1]) < 1e-9:
            uy[-1] = float(y)
        else:
            ux.append(float(x))
            uy.append(float(y))
    return np.interp(grid, np.asarray(ux), np.asarray(uy), left=float("nan"), right=float("nan"))


def _filter_results(
    results: Dict[str, Dict[str, Dict]], tasks: Sequence[str]
) -> Dict[str, Dict[str, Dict]]:
    return {t: results[t] for t in tasks if t in results}


def plot_eval_curves_multi_seed(
    all_results: List[Dict[str, Dict[str, Dict]]],
    out_jpg: Path,
    title_suffix: str = "",
) -> None:
    """Per-seed learning curves (semi-transparent) + bold mean (one panel per task)."""
    if not all_results:
        return
    tasks = list(all_results[0].keys())
    n_seeds = len(all_results)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)

    for ax, env_id in zip(axes[0], tasks):
        for tag, color, label in (
            ("baseline", COLOR_BASE, "fixed hparams"),
            ("meta", COLOR_META, "meta-RL ctrl"),
        ):
            arrays = [
                np.asarray(r[env_id][tag]["episode_returns"], dtype=np.float64)
                for r in all_results
                if env_id in r and tag in r[env_id]
            ]
            if not arrays:
                continue
            n = min(len(a) for a in arrays)
            M = np.stack([a[:n] for a in arrays], axis=0)
            mu = M.mean(0)
            w = max(1, n // 50)
            ep_sm = np.arange(w, n + 1)
            sm_mu = _smooth(mu, w)
            l100 = float(
                np.mean(
                    [
                        r[env_id][tag]["last100_mean"]
                        for r in all_results
                        if env_id in r and tag in r[env_id]
                    ]
                )
            )
            for arr in arrays:
                sm = _smooth(arr[:n], w)
                ax.plot(
                    ep_sm,
                    sm,
                    color=color,
                    alpha=0.35,
                    lw=1.0,
                    zorder=1,
                )
            ax.plot(
                ep_sm,
                sm_mu,
                color=color,
                lw=3.0,
                label=f"{label} mean (last100={l100:.1f})",
                zorder=2,
            )
        ax.set_title(env_id, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episodic return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Meta-HPO vs fixed hparams ({n_seeds} seeds, faint=individual, bold=mean){title_suffix}",
        fontsize=12,
        y=1.02,
    )
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_knob_trajectories_multi_seed(
    all_results: List[Dict[str, Dict[str, Dict]]],
    out_jpg: Path,
) -> None:
    """Per-seed knob trajectories (semi-transparent) + bold mean."""
    if not all_results:
        return
    tasks = list(all_results[0].keys())
    keys = KNOB_PLOT_KEYS
    seed_cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        len(keys), len(tasks), figsize=(4.5 * len(tasks), 2.0 * len(keys)), squeeze=False
    )

    for col, env_id in enumerate(tasks):
        hists = [
            r[env_id]["meta"].get("knob_history", [])
            for r in all_results
            if env_id in r and "meta" in r[env_id] and r[env_id]["meta"].get("knob_history")
        ]
        if not hists:
            continue
        starts = [float(h[0]["step"]) for h in hists]
        ends = [float(h[-1]["step"]) for h in hists]
        lo, hi = max(starts), min(ends)
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, 200)
        for row, key in enumerate(keys):
            ax = axes[row][col]
            usable = [h for h in hists if key in h[0]]
            if not usable:
                ax.set_ylabel(key, fontsize=8)
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center")
                continue
            rows = np.vstack([_interp_hist(h, "step", key, grid) for h in usable])
            mu = np.nanmean(rows, axis=0)
            for i, row_vals in enumerate(rows):
                ax.plot(grid, row_vals, color=seed_cmap(i % 10), alpha=0.35, lw=0.9, zorder=1)
            ax.plot(grid, mu, color=COLOR_META, lw=2.5, zorder=2)
            ax.set_ylabel(key, fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(env_id, fontsize=10)
            if row == len(keys) - 1:
                ax.set_xlabel("Inner step", fontsize=9)

    fig.suptitle(
        f"Hyperparameter trajectories (meta-RL, {len(all_results)} seeds, faint=individual, bold=mean)",
        fontsize=12,
    )
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


def plot_meta_train_progress(history: List[Dict], out_jpg: Path) -> None:
    if not history:
        return
    by_task: Dict[str, List[Tuple[int, float]]] = {}
    for h in history:
        by_task.setdefault(h["env_id"], []).append((h["meta_iter"], h["last100_mean"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.get_cmap("tab10")
    for i, (env_id, pts) in enumerate(sorted(by_task.items())):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=cmap(i % 10), label=env_id, alpha=0.85)
    ax.set_xlabel("Meta-iteration")
    ax.set_ylabel("Inner last-100 return")
    ax.set_title("Meta-training progress across tasks")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_jpg, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"Wrote {out_jpg}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated Gymnasium env ids",
    )
    p.add_argument("--meta-iters", type=int, default=DEFAULT_META_ITERS)
    p.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    p.add_argument(
        "--total-episodes",
        type=int,
        default=None,
        metavar="E",
        help="Stop each inner run (meta-train + eval) after E completed episodes",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for checkpoints/figures (isolates parallel / resume runs)",
    )
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing controller checkpoints and train from scratch",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip meta-train; load controller and run eval vs baseline",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; regenerate figures from eval checkpoints",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run meta-train (no post-hoc eval curves)",
    )
    p.add_argument(
        "--train-missing-only",
        action="store_true",
        help="Skip meta-train if controller checkpoint already exists",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip per-seed figures; write multi-seed aggregate plots only",
    )
    p.add_argument(
        "--reuse-eval-seeds",
        type=str,
        default="",
        help="Comma-separated seeds whose eval checkpoints are reused as-is (skip re-eval)",
    )
    args = p.parse_args()

    tasks = _parse_list(args.tasks)
    seeds = _parse_seeds(args.seeds)
    reuse_eval_seeds = set(_parse_seeds(args.reuse_eval_seeds)) if args.reuse_eval_seeds else set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    run_tag = args.tag
    tag_sfx = f"_{run_tag}" if run_tag else ""
    budget_tag = (
        f"ep{args.total_episodes}"
        if args.total_episodes is not None
        else f"inner{args.inner_steps}"
    )
    if run_tag:
        budget_tag = f"{budget_tag}_{run_tag}"
    all_eval_results: List[Dict[str, Dict[str, Dict]]] = []

    # Episode budget also raises step safety cap for train/eval
    inner_steps = args.inner_steps
    if args.total_episodes is not None:
        inner_steps = max(inner_steps, int(args.total_episodes) * 2000)

    for seed in seeds:
        ckpt_path = CKPT_DIR / f"meta_controller_seed{seed}{tag_sfx}.pt"
        eval_path = CKPT_DIR / f"eval_vs_baseline_seed{seed}{tag_sfx}.pt"

        controller: Optional[MetaHPOController] = None
        history: List[Dict] = []

        if args.plot_only:
            if not eval_path.exists():
                raise FileNotFoundError(f"Missing eval checkpoint: {eval_path}")
            blob = torch.load(eval_path, map_location="cpu", weights_only=False)
            results = _filter_results(blob["results"], tasks)
            all_eval_results.append(results)
            if not args.aggregate_only:
                plot_eval_curves(
                    results,
                    FIG_DIR / f"meta_vs_baseline_seed{seed}{tag_sfx}.jpg",
                    title_suffix=f" (seed={seed}{tag_sfx})",
                )
                plot_knob_trajectories(
                    results, FIG_DIR / f"knob_trajectories_seed{seed}{tag_sfx}.jpg"
                )
                if ckpt_path.exists():
                    train_blob = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                    )
                    plot_meta_train_progress(
                        train_blob.get("history", []),
                        FIG_DIR / f"meta_train_progress_seed{seed}{tag_sfx}.jpg",
                    )
            continue

        if not args.eval_only:
            if args.train_missing_only and ckpt_path.exists():
                blob = torch.load(ckpt_path, map_location=device, weights_only=False)
                done = int(blob.get("meta_iters_done", len(blob.get("history", []))))
                if done >= args.meta_iters:
                    print(
                        f"Controller complete ({done}/{args.meta_iters}), "
                        f"skipping meta-train: {ckpt_path}",
                        flush=True,
                    )
                    controller = MetaHPOController(n_tasks=len(blob["tasks"])).to(device)
                    controller.load_state_dict(blob["controller"])
                    history = blob.get("history", [])
                    tasks = blob.get("tasks", tasks)
                else:
                    train_out = run_meta_train(
                        tasks=tasks,
                        meta_iters=args.meta_iters,
                        inner_steps=inner_steps,
                        seed=seed,
                        device=device,
                        checkpoint_dir=CKPT_DIR,
                        total_episodes=args.total_episodes,
                        tag=run_tag,
                        resume=not args.no_resume,
                    )
                    controller = train_out["controller"]
                    history = train_out["history"]
                    plot_meta_train_progress(
                        history,
                        FIG_DIR / f"meta_train_progress_seed{seed}{tag_sfx}.jpg",
                    )
            else:
                train_out = run_meta_train(
                    tasks=tasks,
                    meta_iters=args.meta_iters,
                    inner_steps=inner_steps,
                    seed=seed,
                    device=device,
                    checkpoint_dir=CKPT_DIR,
                    total_episodes=args.total_episodes,
                    tag=run_tag,
                    resume=not args.no_resume,
                )
                controller = train_out["controller"]
                history = train_out["history"]
                plot_meta_train_progress(
                    history, FIG_DIR / f"meta_train_progress_seed{seed}{tag_sfx}.jpg"
                )
        else:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing controller checkpoint: {ckpt_path}")
            blob = torch.load(ckpt_path, map_location=device, weights_only=False)
            tasks = blob.get("tasks", tasks)
            controller = MetaHPOController(n_tasks=len(tasks)).to(device)
            controller.load_state_dict(blob["controller"])
            history = blob.get("history", [])

        if args.skip_eval or seed in reuse_eval_seeds:
            if seed in reuse_eval_seeds:
                if not eval_path.exists():
                    raise FileNotFoundError(
                        f"--reuse-eval-seeds includes {seed} but missing: {eval_path}"
                    )
                blob = torch.load(eval_path, map_location="cpu", weights_only=False)
                all_eval_results.append(_filter_results(blob["results"], tasks))
                print(f"Reusing eval checkpoint for seed {seed}: {eval_path}", flush=True)
            continue

        results = run_eval_vs_baseline(
            tasks=tasks,
            controller=controller,
            inner_steps=inner_steps,
            seed=seed,
            device=device,
            checkpoint_dir=CKPT_DIR,
            deterministic_ctrl=True,
            total_episodes=args.total_episodes,
            tag=run_tag,
        )
        results = _filter_results(results, tasks)
        if not args.aggregate_only:
            budget_suffix = (
                f"ep={args.total_episodes}"
                if args.total_episodes is not None
                else f"inner={args.inner_steps}"
            )
            plot_eval_curves(
                results,
                FIG_DIR / f"meta_vs_baseline_seed{seed}{tag_sfx}.jpg",
                title_suffix=f" (seed={seed}, {budget_suffix}{tag_sfx})",
            )
            plot_knob_trajectories(
                results, FIG_DIR / f"knob_trajectories_seed{seed}{tag_sfx}.jpg"
            )
        all_eval_results.append(results)

    if all_eval_results and len(all_eval_results) > 1:
        n_seeds_tag = f"{len(all_eval_results)}seeds"
        plot_eval_curves_multi_seed(
            all_eval_results,
            FIG_DIR / f"meta_vs_baseline_{budget_tag}_{n_seeds_tag}.jpg",
            title_suffix=f" ({budget_tag})",
        )
        plot_knob_trajectories_multi_seed(
            all_eval_results,
            FIG_DIR / f"knob_trajectories_{budget_tag}_{n_seeds_tag}.jpg",
        )


if __name__ == "__main__":
    main()
