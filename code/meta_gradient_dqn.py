"""
CleanRL DQN with differentiable, online meta-gradient hyperparameter tuning.

The inner learner follows CleanRL's single-file DQN implementation (replay
buffer, epsilon-greedy behavior, Adam TD updates, and a target network). The
meta-learner optimizes:

  * gamma: the discount used by the DQN training target
  * learning_rate: the Adam learning rate

At each meta-update, the code:

  1. computes a DQN loss on a replay-buffer training batch;
  2. differentiates through one *virtual Adam update*;
  3. evaluates the virtually updated Q-network on an independent replay batch;
  4. backpropagates that held-out loss into gamma and the learning rate.

The held-out target uses a fixed objective discount (`meta_objective_gamma`).
This is important: using the learned gamma in both losses admits the degenerate
solution of changing the target merely to make the TD loss easier.

Example:
    python code/cleanrl_dqn_meta_gradient.py \
        --env-id CartPole-v1 --total-timesteps 500000

Based on CleanRL's cleanrl/dqn.py:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
"""

from __future__ import annotations

import os
import random
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.func import functional_call
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # CleanRL experiment arguments.
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL-meta-gradient"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    # CleanRL DQN arguments.
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    total_episodes: int | None = None
    """stop after this many completed episodes (overrides timestep cap when set)"""
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10

    # Meta-gradient arguments.
    meta_gradient: bool = True
    """if False, use fixed gamma and learning_rate (CleanRL baseline)"""
    meta_learning_rate: float = 1e-3
    meta_update_frequency: int = 100
    meta_batch_size: int = 128
    meta_objective_gamma: float = 0.99
    gamma_min: float = 0.80
    gamma_max: float = 0.9999
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 3e-3
    meta_grad_clip: float = 10.0
    log_interval: int = 100
    returns_output: str | None = None
    """optional JSON path for per-episode returns"""
    tensorboard: bool = True
    """log metrics to TensorBoard"""


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def linear_schedule(
    start_e: float, end_e: float, duration: float, timestep: int
) -> float:
    slope = (end_e - start_e) / max(duration, 1.0)
    return max(slope * timestep + start_e, end_e)


class ReplayBatch(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Minimal equivalent of the CleanRL/SB3 replay buffer for one environment."""

    def __init__(
        self,
        capacity: int,
        observation_space: gym.spaces.Space,
        device: torch.device,
    ):
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("This DQN supports Box observations and Discrete actions.")
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.full = False
        shape = observation_space.shape
        self.observations = np.zeros((capacity, *shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def add(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.next_observations[self.position] = next_observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            self.full = True

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = np.random.randint(0, len(self), size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(
                self.observations[indices], device=self.device
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=self.device
            ),
            actions=torch.as_tensor(
                self.actions[indices, None], device=self.device
            ),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
        )


class QNetwork(nn.Module):
    """The network architecture from CleanRL's classic-control DQN."""

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        observation_size = int(np.prod(envs.single_observation_space.shape))
        self.network = nn.Sequential(
            nn.Linear(observation_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, envs.single_action_space.n),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.reshape(observation.shape[0], -1))


def _inverse_bounded_sigmoid(value: float, lower: float, upper: float) -> float:
    if not lower < value < upper:
        raise ValueError(f"Initial value {value} must be inside ({lower}, {upper}).")
    probability = (value - lower) / (upper - lower)
    return float(np.log(probability / (1.0 - probability)))


class MetaHyperparameters(nn.Module):
    """Unconstrained parameters transformed into valid DQN hyperparameters."""

    def __init__(self, args: Args, device: torch.device):
        super().__init__()
        self.gamma_min = args.gamma_min
        self.gamma_max = args.gamma_max
        self.lr_min = args.learning_rate_min
        self.lr_max = args.learning_rate_max
        self.raw_gamma = nn.Parameter(
            torch.tensor(
                _inverse_bounded_sigmoid(
                    args.gamma, args.gamma_min, args.gamma_max
                ),
                dtype=torch.float32,
                device=device,
            )
        )
        self.raw_learning_rate = nn.Parameter(
            torch.tensor(
                _inverse_bounded_sigmoid(
                    args.learning_rate,
                    args.learning_rate_min,
                    args.learning_rate_max,
                ),
                dtype=torch.float32,
                device=device,
            )
        )

    def gamma(self) -> torch.Tensor:
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(
            self.raw_gamma
        )

    def learning_rate(self) -> torch.Tensor:
        return self.lr_min + (self.lr_max - self.lr_min) * torch.sigmoid(
            self.raw_learning_rate
        )


def td_loss(
    q_values: torch.Tensor,
    target_network: QNetwork,
    batch: ReplayBatch,
    gamma: torch.Tensor | float,
) -> torch.Tensor:
    """DQN TD loss; gradients may flow through gamma and q_values."""
    with torch.no_grad():
        target_max = target_network(batch.next_observations).max(dim=1).values
    target = batch.rewards + gamma * target_max * (1.0 - batch.dones)
    prediction = q_values.gather(1, batch.actions).squeeze(1)
    return F.mse_loss(prediction, target)


def virtual_adam_parameters(
    named_parameters: dict[str, nn.Parameter],
    gradients: tuple[torch.Tensor, ...],
    optimizer: optim.Adam,
    learning_rate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply Adam functionally without mutating its parameters or state."""
    group = optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    epsilon = group["eps"]
    weight_decay = group["weight_decay"]
    maximize = group["maximize"]
    if group.get("amsgrad", False):
        raise ValueError("The virtual update does not support AMSGrad.")

    adapted: dict[str, torch.Tensor] = {}
    for (name, parameter), gradient in zip(named_parameters.items(), gradients):
        if maximize:
            gradient = -gradient
        if weight_decay:
            gradient = gradient + weight_decay * parameter

        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter)).detach()
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter)).detach()
        old_step = state.get("step", 0)
        if isinstance(old_step, torch.Tensor):
            old_step = int(old_step.item())
        step = int(old_step) + 1

        next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
        next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
        exp_avg_hat = next_exp_avg / (1.0 - beta1**step)
        exp_avg_sq_hat = next_exp_avg_sq / (1.0 - beta2**step)
        # Adam's sqrt(v) is finite at v=0, but its derivative is not. The
        # epsilon inside sqrt keeps the second-order meta-gradient finite.
        denominator = (exp_avg_sq_hat + epsilon**2).sqrt() + epsilon
        update = exp_avg_hat / denominator
        adapted[name] = parameter - learning_rate * update
    return adapted


def meta_gradient_step(
    q_network: QNetwork,
    target_network: QNetwork,
    dqn_optimizer: optim.Adam,
    meta_parameters: MetaHyperparameters,
    meta_optimizer: optim.Optimizer,
    train_batch: ReplayBatch,
    validation_batch: ReplayBatch,
    objective_gamma: float,
    grad_clip: float,
) -> tuple[float, float, float]:
    """Differentiate held-out TD loss through one virtual CleanRL Adam update."""
    named_parameters = dict(q_network.named_parameters())
    inner_loss = td_loss(
        q_network(train_batch.observations),
        target_network,
        train_batch,
        meta_parameters.gamma(),
    )
    inner_gradients = torch.autograd.grad(
        inner_loss, tuple(named_parameters.values()), create_graph=True
    )
    adapted_parameters = virtual_adam_parameters(
        named_parameters,
        inner_gradients,
        dqn_optimizer,
        meta_parameters.learning_rate(),
    )

    validation_q = functional_call(
        q_network, adapted_parameters, (validation_batch.observations,)
    )
    outer_loss = td_loss(
        validation_q,
        target_network,
        validation_batch,
        objective_gamma,
    )

    meta_optimizer.zero_grad()
    meta_parameter_tuple = tuple(meta_parameters.parameters())
    meta_gradients = torch.autograd.grad(outer_loss, meta_parameter_tuple)
    if any(not torch.isfinite(gradient).all() for gradient in meta_gradients):
        raise FloatingPointError(
            "Non-finite meta-gradient. Reduce --meta-learning-rate or narrow "
            "the hyperparameter bounds."
        )
    for parameter, gradient in zip(meta_parameter_tuple, meta_gradients):
        parameter.grad = gradient
    gamma_grad = float(meta_gradients[0].detach().item())
    lr_grad = float(meta_gradients[1].detach().item())
    nn.utils.clip_grad_norm_(meta_parameters.parameters(), grad_clip)
    meta_optimizer.step()
    return float(outer_loss.detach().item()), gamma_grad, lr_grad


def run_training(args: Args) -> dict:
    if args.num_envs != 1:
        raise ValueError("CleanRL classic-control DQN currently supports num_envs=1.")
    if args.meta_gradient and args.meta_update_frequency % args.train_frequency != 0:
        raise ValueError(
            "meta_update_frequency must be divisible by train_frequency."
        )
    if not 0.0 <= args.meta_objective_gamma <= 1.0:
        raise ValueError("meta_objective_gamma must be in [0, 1].")

    timestep_limit = args.total_timesteps
    if args.total_episodes is not None:
        timestep_limit = max(args.total_timesteps, args.total_episodes * 2_000)

    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
    writer = None
    if args.tensorboard:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n"
            + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + index,
                index,
                args.capture_video,
                run_name,
            )
            for index in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_observation_space, gym.spaces.Box):
        raise TypeError("Only Box observation spaces are supported.")
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("Only Discrete action spaces are supported.")

    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    dqn_optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    meta_parameters = MetaHyperparameters(args, device)
    meta_optimizer = optim.Adam(
        meta_parameters.parameters(), lr=args.meta_learning_rate
    )
    replay_buffer = ReplayBuffer(
        args.buffer_size, envs.single_observation_space, device
    )

    start_time = time.time()
    observation, _ = envs.reset(seed=args.seed)
    episode_returns: list[float] = []
    training_done = False
    for global_step in range(timestep_limit):
        if training_done:
            break
        if args.total_episodes is not None:
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_episodes,
                len(episode_returns),
            )
        else:
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * timestep_limit,
                global_step,
            )
        if random.random() < epsilon:
            actions = np.array(
                [
                    envs.single_action_space.sample()
                    for _ in range(envs.num_envs)
                ]
            )
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    )
                )
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_observation, rewards, terminations, truncations, infos = envs.step(
            actions
        )
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = float(np.asarray(info["episode"]["r"]).item())
                    episodic_length = int(np.asarray(info["episode"]["l"]).item())
                    episode_returns.append(episodic_return)
                    print(
                        f"episode={len(episode_returns)}, global_step={global_step}, "
                        f"episodic_return={episodic_return}"
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "charts/episodic_return", episodic_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", episodic_length, global_step
                        )
                    if (
                        args.total_episodes is not None
                        and len(episode_returns) >= args.total_episodes
                    ):
                        training_done = True
                        break

        real_next_observation = next_observation.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                real_next_observation[index] = infos["final_observation"][index]
        replay_buffer.add(
            observation[0],
            real_next_observation[0],
            int(actions[0]),
            float(rewards[0]),
            bool(terminations[0]),
        )
        observation = next_observation

        if (
            global_step > args.learning_starts
            and global_step % args.train_frequency == 0
        ):
            train_batch = replay_buffer.sample(args.batch_size)

            if (
                args.meta_gradient
                and global_step % args.meta_update_frequency == 0
            ):
                validation_batch = replay_buffer.sample(args.meta_batch_size)
                meta_loss, gamma_grad, lr_grad = meta_gradient_step(
                    q_network,
                    target_network,
                    dqn_optimizer,
                    meta_parameters,
                    meta_optimizer,
                    train_batch,
                    validation_batch,
                    args.meta_objective_gamma,
                    args.meta_grad_clip,
                )
                if writer is not None:
                    writer.add_scalar(
                        "meta/validation_td_loss", meta_loss, global_step
                    )
                    writer.add_scalar("meta/raw_gamma_grad", gamma_grad, global_step)
                    writer.add_scalar(
                        "meta/raw_learning_rate_grad", lr_grad, global_step
                    )

            if args.meta_gradient:
                learned_gamma = float(meta_parameters.gamma().detach().item())
                learned_lr = float(meta_parameters.learning_rate().detach().item())
            else:
                learned_gamma = args.gamma
                learned_lr = args.learning_rate
            dqn_optimizer.param_groups[0]["lr"] = learned_lr
            loss = td_loss(
                q_network(train_batch.observations),
                target_network,
                train_batch,
                learned_gamma,
            )
            dqn_optimizer.zero_grad()
            loss.backward()
            dqn_optimizer.step()

            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    old_values = q_network(train_batch.observations).gather(
                        1, train_batch.actions
                    )
                if writer is not None:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar(
                        "losses/q_values", old_values.mean().item(), global_step
                    )
                    writer.add_scalar("meta/gamma", learned_gamma, global_step)
                    writer.add_scalar("meta/learning_rate", learned_lr, global_step)
                sps = int(global_step / max(time.time() - start_time, 1e-9))
                print(
                    f"SPS={sps}, gamma={learned_gamma:.6f}, "
                    f"learning_rate={learned_lr:.7f}"
                )
                if writer is not None:
                    writer.add_scalar("charts/SPS", sps, global_step)

        if global_step % args.target_network_frequency == 0:
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_parameter.copy_(
                        args.tau * parameter
                        + (1.0 - args.tau) * target_parameter
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "q_network": q_network.state_dict(),
                "meta_parameters": meta_parameters.state_dict(),
                "gamma": float(meta_parameters.gamma().detach().item()),
                "learning_rate": float(
                    meta_parameters.learning_rate().detach().item()
                ),
                "args": vars(args),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    if writer is not None:
        writer.close()

    results = {
        "episode_returns": episode_returns,
        "meta_gradient": args.meta_gradient,
        "seed": args.seed,
        "env_id": args.env_id,
        "total_episodes": args.total_episodes,
        "global_steps": global_step + 1,
        "final_gamma": float(meta_parameters.gamma().detach().item())
        if args.meta_gradient
        else args.gamma,
        "final_learning_rate": float(
            meta_parameters.learning_rate().detach().item()
        )
        if args.meta_gradient
        else args.learning_rate,
    }
    if args.returns_output:
        output_path = Path(args.returns_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return results


def main() -> None:
    args = tyro.cli(Args)
    run_training(args)


if __name__ == "__main__":
    main()
