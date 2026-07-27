"""
Multi-task meta-RL hyperparameter controller for DQN.

Outer loop: sample a task, keep a shared GRU controller.
Inner loop: fresh DQN per task; controller samples major training knobs
            (lr_mult, eps_end, train_freq, target_freq) applied to Adam /
            schedules (never via loss scaling). Controller trained with
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
from torch.distributions import Categorical, Normal

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
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
STEP_COST = 0.01
RETURN_EMA_ALPHA = 0.1
META_REWARD_CLIP = 5.0  # clip Δreturn reward for PG stability across tasks

TRAIN_FREQ_CHOICES = [1, 2, 4, 8, 10]
TARGET_FREQ_CHOICES = [1, 50, 100, 250, 500]

LR_MULT_LO, LR_MULT_HI = 0.1, 3.0
EPS_END_LO, EPS_END_HI = 0.01, 0.2

COLOR_BASE = "#2ca02c"
COLOR_META = "#ff7f0e"

# Meta-state layout (no task emb):
#   loss_ema, grad_ema, return_ema, eps, progress, lr_mult, train_freq_n, target_freq_n
BASE_STATE_DIM = 8


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
        train_freq: int,
        target_freq: int,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
    ):
        self.lr_mult = lr_mult
        self.eps_end = eps_end
        self.train_freq = train_freq
        self.target_freq = target_freq
        self.log_prob = log_prob
        self.entropy = entropy
        self.value = value


class MetaHPOController(nn.Module):
    """
    GRU policy: continuous lr_mult / eps_end + categorical train/target freqs.
    State = base features + task embedding.
    """

    def __init__(self, n_tasks: int, state_dim: int = BASE_STATE_DIM):
        super().__init__()
        self.n_tasks = n_tasks
        self.state_dim = state_dim
        self.task_emb = nn.Embedding(n_tasks, TASK_EMB_DIM)
        in_dim = state_dim + TASK_EMB_DIM
        self.gru = nn.GRUCell(in_dim, CTRL_HIDDEN)
        self.trunk = nn.Sequential(nn.Linear(CTRL_HIDDEN, CTRL_HIDDEN), nn.ReLU())
        # Continuous: raw params → (mu_lr, log_std_lr, mu_eps, log_std_eps)
        self.cont_head = nn.Linear(CTRL_HIDDEN, 4)
        self.tf_head = nn.Linear(CTRL_HIDDEN, len(TRAIN_FREQ_CHOICES))
        self.tgt_head = nn.Linear(CTRL_HIDDEN, len(TARGET_FREQ_CHOICES))
        self.value_head = nn.Linear(CTRL_HIDDEN, 1)
        self._init_biases()

    def _init_biases(self) -> None:
        with torch.no_grad():
            # Prefer near-baseline at start: lr_mult≈1, eps_end≈0.05
            # softplus(0.5413)≈1 → map via sigmoid later for bounds
            self.cont_head.bias.zero_()
            self.cont_head.bias[0] = 0.0  # logit for mid lr after transform
            self.cont_head.bias[1] = -1.0  # small log_std
            self.cont_head.bias[2] = -1.0  # eps toward lower end of range
            self.cont_head.bias[3] = -1.0
            nn.init.zeros_(self.cont_head.weight)
            # Prefer default train_freq=10 (last), target_freq=500 (last)
            self.tf_head.bias.zero_()
            self.tf_head.bias[-1] = 1.0
            self.tgt_head.bias.zero_()
            self.tgt_head.bias[-1] = 1.0
            nn.init.zeros_(self.tf_head.weight)
            nn.init.zeros_(self.tgt_head.weight)

    def zero_hidden(
        self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(batch, CTRL_HIDDEN, device=device, dtype=dtype)

    @staticmethod
    def _squash_lr(u: torch.Tensor) -> torch.Tensor:
        # Map unbounded → [LR_MULT_LO, LR_MULT_HI] via sigmoid
        return LR_MULT_LO + (LR_MULT_HI - LR_MULT_LO) * torch.sigmoid(u)

    @staticmethod
    def _squash_eps(u: torch.Tensor) -> torch.Tensor:
        return EPS_END_LO + (EPS_END_HI - EPS_END_LO) * torch.sigmoid(u)

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

        cont = self.cont_head(feat)
        mu_lr, log_std_lr = cont[:, 0], cont[:, 1].clamp(-3.0, 0.5)
        mu_eps, log_std_eps = cont[:, 2], cont[:, 3].clamp(-3.0, 0.5)
        std_lr = log_std_lr.exp()
        std_eps = log_std_eps.exp()

        dist_lr = Normal(mu_lr, std_lr)
        dist_eps = Normal(mu_eps, std_eps)
        dist_tf = Categorical(logits=self.tf_head(feat))
        dist_tgt = Categorical(logits=self.tgt_head(feat))

        if deterministic:
            u_lr = mu_lr
            u_eps = mu_eps
            tf_idx = dist_tf.probs.argmax(dim=-1)
            tgt_idx = dist_tgt.probs.argmax(dim=-1)
        else:
            u_lr = dist_lr.rsample()
            u_eps = dist_eps.rsample()
            tf_idx = dist_tf.sample()
            tgt_idx = dist_tgt.sample()

        lr_t = self._squash_lr(u_lr)
        eps_t = self._squash_eps(u_eps)

        # Change-of-variable ignored for simplicity (bounded via sigmoid of sample)
        log_prob = (
            dist_lr.log_prob(u_lr)
            + dist_eps.log_prob(u_eps)
            + dist_tf.log_prob(tf_idx)
            + dist_tgt.log_prob(tgt_idx)
        )
        entropy = (
            dist_lr.entropy()
            + dist_eps.entropy()
            + dist_tf.entropy()
            + dist_tgt.entropy()
        )
        value = self.value_head(feat).squeeze(-1)

        train_freq = TRAIN_FREQ_CHOICES[int(tf_idx[0].item())]
        target_freq = TARGET_FREQ_CHOICES[int(tgt_idx[0].item())]
        # For batch>1 use first for scalar application in single-env loop
        if state.shape[0] > 1:
            train_freq = TRAIN_FREQ_CHOICES[int(tf_idx[0].item())]
            target_freq = TARGET_FREQ_CHOICES[int(tgt_idx[0].item())]

        action = MetaAction(
            lr_mult=float(lr_t[0].item()),
            eps_end=float(eps_t[0].item()),
            train_freq=train_freq,
            target_freq=target_freq,
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
    train_freq: int,
    target_freq: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize knobs to roughly [0, 1] scales."""
    tf_n = TRAIN_FREQ_CHOICES.index(train_freq) / max(1, len(TRAIN_FREQ_CHOICES) - 1)
    tgt_n = TARGET_FREQ_CHOICES.index(target_freq) / max(1, len(TARGET_FREQ_CHOICES) - 1)
    vec = np.array(
        [
            np.tanh(loss_ema),  # soft-bound loss
            np.tanh(grad_ema / 10.0),
            np.tanh(return_ema / 100.0),
            float(eps),
            float(np.clip(progress, 0.0, 1.0)),
            (lr_mult - LR_MULT_LO) / (LR_MULT_HI - LR_MULT_LO),
            tf_n,
            tgt_n,
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
        train_freq = int(fixed_knobs.get("train_freq", DEFAULT_TRAIN_FREQ))
        target_freq = int(fixed_knobs.get("target_freq", DEFAULT_TARGET_FREQ))
    else:
        lr_mult, eps_end = 1.0, END_E
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

    obs, _ = env.reset(seed=seed)
    t = 0
    while t < inner_steps:
        n_ep = len(episode_returns)
        eps = _linear_schedule(START_E, eps_end, epsilon_decay_episodes, n_ep)
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

            progress = t / max(1, inner_steps)
            state = build_meta_state(
                loss_ema,
                grad_ema,
                return_ema,
                eps,
                progress,
                lr_mult,
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
) -> Dict:
    _set_seed(seed)
    n_tasks = len(tasks)
    controller = MetaHPOController(n_tasks=n_tasks).to(device)
    opt_ctrl = optim.Adam(controller.parameters(), lr=CTRL_LR)

    history: List[Dict] = []
    per_task_curves: Dict[str, List[List[float]]] = {t: [] for t in tasks}

    print("=" * 60, flush=True)
    print("confgate_meta_hpo — multi-task meta-RL HPO", flush=True)
    print(f"  Tasks       : {list(tasks)}", flush=True)
    print(f"  Meta iters  : {meta_iters}", flush=True)
    print(f"  Inner steps : {inner_steps:,}", flush=True)
    print(f"  Seed        : {seed}", flush=True)
    print(f"  Device      : {device}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.perf_counter()
    # Round-robin with a shuffled order each full cycle (guarantees all tasks)
    task_order = list(range(n_tasks))
    random.shuffle(task_order)
    for meta_i in range(meta_iters):
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

    elapsed = time.perf_counter() - t0
    print(f"Meta-train done in {elapsed:.1f}s", flush=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"meta_controller_seed{seed}.pt"
    torch.save(
        {
            "controller": controller.state_dict(),
            "tasks": list(tasks),
            "seed": seed,
            "meta_iters": meta_iters,
            "inner_steps": inner_steps,
            "history": history,
            "per_task_curves": per_task_curves,
        },
        ckpt_path,
    )
    print(f"  Checkpoint: {ckpt_path}", flush=True)

    # Also dump JSON summary (no huge curves)
    summary_path = checkpoint_dir / f"meta_train_summary_seed{seed}.json"
    summary = {
        "seed": seed,
        "tasks": list(tasks),
        "meta_iters": meta_iters,
        "inner_steps": inner_steps,
        "history": [
            {
                k: v
                for k, v in h.items()
                if k != "knob_history"
            }
            for h in history
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "controller": controller,
        "history": history,
        "per_task_curves": per_task_curves,
        "ckpt_path": ckpt_path,
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
) -> Dict[str, Dict[str, Dict]]:
    """
    For each task, run fixed-hparam baseline and (if controller) meta-controlled
    inner training. Meta eval does NOT update the controller.
    """
    results: Dict[str, Dict[str, Dict]] = {t: {} for t in tasks}
    fixed = {
        "lr_mult": 1.0,
        "eps_end": END_E,
        "train_freq": DEFAULT_TRAIN_FREQ,
        "target_freq": DEFAULT_TARGET_FREQ,
    }

    if controller is None:
        controller = MetaHPOController(n_tasks=len(tasks)).to(device)
    # lr=0 so accidental step is a no-op; update_controller=False skips backward
    frozen_opt = optim.Adam(controller.parameters(), lr=0.0)

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
        )
        results[env_id]["meta"] = meta

        print(
            f"  {env_id}: baseline last100={base['last100_mean']:.1f}"
            f"  meta last100={meta['last100_mean']:.1f}",
            flush=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_path = checkpoint_dir / f"eval_vs_baseline_seed{seed}.pt"
    serializable = {
        env_id: {
            tag: {
                "episode_returns": res["episode_returns"],
                "last100_mean": res["last100_mean"],
                "knob_history": res.get("knob_history", []),
            }
            for tag, res in tags.items()
        }
        for env_id, tags in results.items()
    }
    torch.save({"seed": seed, "tasks": list(tasks), "results": serializable}, eval_path)
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

    keys = ["lr_mult", "eps_end", "train_freq", "target_freq"]
    fig, axes = plt.subplots(
        len(keys), len(rows), figsize=(4.5 * len(rows), 2.2 * len(keys)), squeeze=False
    )
    for col, (env_id, hist) in enumerate(rows):
        steps = [h["step"] for h in hist]
        for row, key in enumerate(keys):
            ax = axes[row][col]
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
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
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
    args = p.parse_args()

    tasks = _parse_list(args.tasks)
    seeds = _parse_seeds(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        ckpt_path = CKPT_DIR / f"meta_controller_seed{seed}.pt"
        eval_path = CKPT_DIR / f"eval_vs_baseline_seed{seed}.pt"

        controller: Optional[MetaHPOController] = None
        history: List[Dict] = []

        if args.plot_only:
            if not eval_path.exists():
                raise FileNotFoundError(f"Missing eval checkpoint: {eval_path}")
            blob = torch.load(eval_path, map_location="cpu", weights_only=False)
            results = blob["results"]
            # wrap into expected shape for plotting
            plot_eval_curves(
                results,
                FIG_DIR / f"meta_vs_baseline_seed{seed}.jpg",
                title_suffix=f" (seed={seed})",
            )
            plot_knob_trajectories(
                results, FIG_DIR / f"knob_trajectories_seed{seed}.jpg"
            )
            if ckpt_path.exists():
                train_blob = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                plot_meta_train_progress(
                    train_blob.get("history", []),
                    FIG_DIR / f"meta_train_progress_seed{seed}.jpg",
                )
            continue

        if not args.eval_only:
            if args.train_missing_only and ckpt_path.exists():
                print(f"Controller present, skipping meta-train: {ckpt_path}", flush=True)
                blob = torch.load(ckpt_path, map_location=device, weights_only=False)
                controller = MetaHPOController(n_tasks=len(blob["tasks"])).to(device)
                controller.load_state_dict(blob["controller"])
                history = blob.get("history", [])
                tasks = blob.get("tasks", tasks)
            else:
                train_out = run_meta_train(
                    tasks=tasks,
                    meta_iters=args.meta_iters,
                    inner_steps=args.inner_steps,
                    seed=seed,
                    device=device,
                    checkpoint_dir=CKPT_DIR,
                )
                controller = train_out["controller"]
                history = train_out["history"]
                plot_meta_train_progress(
                    history, FIG_DIR / f"meta_train_progress_seed{seed}.jpg"
                )
        else:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing controller checkpoint: {ckpt_path}")
            blob = torch.load(ckpt_path, map_location=device, weights_only=False)
            tasks = blob.get("tasks", tasks)
            controller = MetaHPOController(n_tasks=len(tasks)).to(device)
            controller.load_state_dict(blob["controller"])
            history = blob.get("history", [])

        if args.skip_eval:
            continue

        results = run_eval_vs_baseline(
            tasks=tasks,
            controller=controller,
            inner_steps=args.inner_steps,
            seed=seed,
            device=device,
            checkpoint_dir=CKPT_DIR,
            deterministic_ctrl=True,
        )
        plot_eval_curves(
            results,
            FIG_DIR / f"meta_vs_baseline_seed{seed}.jpg",
            title_suffix=f" (seed={seed}, inner={args.inner_steps})",
        )
        plot_knob_trajectories(
            results, FIG_DIR / f"knob_trajectories_seed{seed}.jpg"
        )


if __name__ == "__main__":
    main()
