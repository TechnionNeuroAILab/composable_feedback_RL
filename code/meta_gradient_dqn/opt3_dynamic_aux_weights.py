"""
Opt 3: Dynamic auxiliary outer-loss weights for confidence meta-gradient DQN.

Benchmark variant isolating dynamic outer-loss adaptation: the meta-controller
predicts auxiliary_weight alongside gamma and learning rate.

Example:
    python code/opt3_dynamic_aux_weights.py \
        --env-id CartPole-v1 --total-timesteps 500000
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Optional

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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL-meta-gradient-confidence"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    tensorboard: bool = True
    save_model: bool = False
    output_json: Optional[str] = None

    # CleanRL DQN settings.
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    total_episodes: int | None = None
    """stop after this many completed episodes"""
    learning_rate: float = 2.5e-4
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

    # Confidence-conditioned meta-gradient settings.
    meta_gradient: bool = True
    meta_learning_rate: float = 1e-3
    meta_update_frequency: int = 100
    meta_batch_size: int = 128
    meta_objective_gamma: float = 0.99
    meta_gradient_clip: float = 10.0
    gamma_min: float = 0.80
    gamma_max: float = 0.9999
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 3e-3
    auxiliary_weight_min: float = 0.10
    auxiliary_weight_max: float = 2.0
    initial_auxiliary_weight: float = 0.25
    outer_partial_coefficient: float = 0.25
    consistency_coefficient: float = 0.05
    worst_view_coefficient: float = 0.10
    worst_view_temperature: float = 0.25
    telemetry_ema_alpha: float = 0.05
    log_interval: int = 1_000


class ReplayBatch(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_space: gym.spaces.Box,
        device: torch.device,
    ) -> None:
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


def make_env(
    env_id: str,
    seed: int,
    capture_video: bool,
    run_name: str,
):
    def thunk():
        if capture_video:
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


class ConfidenceQNetwork(nn.Module):
    """Shared encoder and full/cart/pole action-value heads."""

    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        observation_size = int(np.prod(envs.single_observation_space.shape))
        if observation_size != 4:
            raise ValueError(
                "This confidence decomposition requires a four-dimensional "
                "CartPole-style observation."
            )
        action_count = envs.single_action_space.n
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.full_head = nn.Linear(84, action_count)
        self.cart_head = nn.Linear(84, action_count)
        self.pole_head = nn.Linear(84, action_count)

    @staticmethod
    def partial_observations(
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cart_observation = observation.clone()
        cart_observation[..., 2:4] = 0.0
        pole_observation = observation.clone()
        pole_observation[..., 0:2] = 0.0
        return cart_observation, pole_observation

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = observation.float().reshape(observation.shape[0], -1)
        cart_observation, pole_observation = self.partial_observations(observation)
        q_full = self.full_head(self.encoder(observation))
        q_cart = self.cart_head(self.encoder(cart_observation))
        q_pole = self.pole_head(self.encoder(pole_observation))
        return q_full, q_cart, q_pole

    def main_parameters(self) -> list[nn.Parameter]:
        """Parameters updated by the inner DQN loss (full head + shared encoder)."""
        return list(self.encoder.parameters()) + list(self.full_head.parameters())

    def main_named_parameters(self) -> Dict[str, nn.Parameter]:
        prefix_blocklist = ("cart_head.", "pole_head.")
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if not name.startswith(prefix_blocklist)
        }


class ControllerOutputs(NamedTuple):
    gamma: torch.Tensor
    learning_rate: torch.Tensor
    auxiliary_weight: torch.Tensor


def inverse_bounded_sigmoid(value: float, lower: float, upper: float) -> float:
    if not lower < value < upper:
        raise ValueError(f"{value} must be strictly inside ({lower}, {upper})")
    probability = (value - lower) / (upper - lower)
    return math.log(probability / (1.0 - probability))


class ConfidenceMetaController(nn.Module):
    """Maps DQN confidence and training telemetry to bounded hyperparameters."""

    STATE_DIM = 19

    def __init__(self, args: Args):
        super().__init__()
        self.gamma_min = args.gamma_min
        self.gamma_max = args.gamma_max
        self.lr_min = args.learning_rate_min
        self.lr_max = args.learning_rate_max
        self.aux_min = args.auxiliary_weight_min
        self.aux_max = args.auxiliary_weight_max
        self.network = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

        gamma_bias = inverse_bounded_sigmoid(
            args.gamma, args.gamma_min, args.gamma_max
        )
        lr_bias = inverse_bounded_sigmoid(
            args.learning_rate,
            args.learning_rate_min,
            args.learning_rate_max,
        )
        aux_bias = inverse_bounded_sigmoid(
            args.initial_auxiliary_weight,
            args.auxiliary_weight_min,
            args.auxiliary_weight_max,
        )
        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias.copy_(torch.tensor([gamma_bias, lr_bias, aux_bias]))

    def forward(self, state: torch.Tensor) -> ControllerOutputs:
        raw = self.network(state).squeeze(0)
        fractions = torch.sigmoid(raw)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * fractions[0]
        learning_rate = self.lr_min + (self.lr_max - self.lr_min) * fractions[1]
        auxiliary_weight = self.aux_min + (
            self.aux_max - self.aux_min
        ) * fractions[2]
        return ControllerOutputs(gamma, learning_rate, auxiliary_weight)


def q_statistics(q_values: torch.Tensor) -> list[torch.Tensor]:
    sorted_values = q_values.sort(dim=1, descending=True).values
    margin = (sorted_values[:, 0] - sorted_values[:, 1]).mean()
    return [q_values.mean(), q_values.std(unbiased=False), margin]


def normalized_log_learning_rate(
    learning_rate: float, minimum: float, maximum: float
) -> float:
    return float(
        (math.log(learning_rate) - math.log(minimum))
        / (math.log(maximum) - math.log(minimum))
    )


def build_controller_state(
    q_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    loss_ema: float,
    grad_ema: float,
    return_ema: float,
    epsilon: float,
    progress: float,
    current_gamma: float,
    current_learning_rate: float,
    args: Args,
) -> torch.Tensor:
    """Build a detached, normalized 19-dimensional meta-state."""
    q_full, q_cart, q_pole = (q.detach() for q in q_values)
    values: list[torch.Tensor] = []
    for q_value in (q_full, q_cart, q_pole):
        values.extend(q_statistics(q_value))
    values.extend(
        [
            (q_full - q_cart).abs().mean(),
            (q_full - q_pole).abs().mean(),
            (q_cart - q_pole).abs().mean(),
        ]
    )
    device = q_full.device
    scalar_values = torch.tensor(
        [
            math.tanh(loss_ema),
            math.tanh(grad_ema / 10.0),
            math.tanh(return_ema / 100.0),
            epsilon,
            min(max(progress, 0.0), 1.0),
            (current_gamma - args.gamma_min)
            / (args.gamma_max - args.gamma_min),
            normalized_log_learning_rate(
                current_learning_rate,
                args.learning_rate_min,
                args.learning_rate_max,
            ),
        ],
        dtype=torch.float32,
        device=device,
    )
    state = torch.cat([torch.stack(values), scalar_values])
    if state.numel() != ConfidenceMetaController.STATE_DIM:
        raise RuntimeError(f"unexpected controller state size: {state.numel()}")
    # Compress unbounded Q statistics and disagreements.
    state[:12] = torch.tanh(state[:12] / 10.0)
    return state.unsqueeze(0)


def per_view_td_losses(
    q_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_network: ConfidenceQNetwork,
    batch: ReplayBatch,
    gamma: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target_q_values = target_network(batch.next_observations)
        target_max_values = tuple(
            target_q.max(dim=1).values for target_q in target_q_values
        )
    # Keep gamma outside no_grad: the inner TD targets must expose a gradient
    # path from the virtual DQN update back into the meta-controller.
    targets = tuple(
        batch.rewards + gamma * target_max * (1.0 - batch.dones)
        for target_max in target_max_values
    )
    losses = []
    for q_value, target in zip(q_values, targets):
        prediction = q_value.gather(1, batch.actions).squeeze(1)
        losses.append(F.mse_loss(prediction, target))
    return losses[0], losses[1], losses[2]


def inner_loss(
    q_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_network: ConfidenceQNetwork,
    batch: ReplayBatch,
    gamma: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-head TD loss only; auxiliary heads are not part of training."""
    full_loss, _, _ = per_view_td_losses(
        q_values, target_network, batch, gamma
    )
    return full_loss, full_loss


def robust_outer_loss(
    q_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_network: ConfidenceQNetwork,
    batch: ReplayBatch,
    args: Args,
    outputs: ControllerOutputs,
) -> torch.Tensor:
    losses = per_view_td_losses(
        q_values, target_network, batch, args.meta_objective_gamma
    )
    view_losses = torch.stack(losses)
    temperature = max(args.worst_view_temperature, 1e-6)
    worst_view = temperature * torch.logsumexp(
        view_losses / temperature, dim=0
    )
    q_full, q_cart, q_pole = q_values
    consistency = 0.5 * (
        F.smooth_l1_loss(q_full, q_cart)
        + F.smooth_l1_loss(q_full, q_pole)
    )
    auxiliary_terms = (
        args.outer_partial_coefficient * (losses[1] + losses[2])
        + args.consistency_coefficient * consistency
        + args.worst_view_coefficient * worst_view
    )
    return losses[0] + outputs.auxiliary_weight * auxiliary_terms


def virtual_adam_parameters(
    named_parameters: Dict[str, nn.Parameter],
    gradients: tuple[torch.Tensor, ...],
    optimizer: optim.Adam,
    learning_rate: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Apply one differentiable Adam update without mutating real state."""
    group = optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    epsilon = group["eps"]
    weight_decay = group["weight_decay"]
    if group.get("amsgrad", False):
        raise ValueError("The virtual update does not support AMSGrad.")

    adapted: Dict[str, torch.Tensor] = {}
    for (name, parameter), gradient in zip(named_parameters.items(), gradients):
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter)).detach()
        exp_avg_sq = state.get(
            "exp_avg_sq", torch.zeros_like(parameter)
        ).detach()
        old_step = state.get("step", 0)
        if isinstance(old_step, torch.Tensor):
            old_step = int(old_step.item())
        step = int(old_step) + 1
        if weight_decay:
            gradient = gradient + weight_decay * parameter
        next_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
        next_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
        average_hat = next_avg / (1.0 - beta1**step)
        square_hat = next_sq / (1.0 - beta2**step)
        denominator = (square_hat + epsilon**2).sqrt() + epsilon
        adapted[name] = parameter - learning_rate * average_hat / denominator
    return adapted


def meta_gradient_step(
    q_network: ConfidenceQNetwork,
    target_network: ConfidenceQNetwork,
    dqn_optimizer: optim.Adam,
    controller: ConfidenceMetaController,
    meta_optimizer: optim.Optimizer,
    train_batch: ReplayBatch,
    validation_batch: ReplayBatch,
    controller_state: torch.Tensor,
    args: Args,
) -> tuple[float, ControllerOutputs, float]:
    outputs = controller(controller_state)
    named_parameters = q_network.main_named_parameters()
    train_q_values = q_network(train_batch.observations)
    training_loss, _ = inner_loss(
        train_q_values, target_network, train_batch, outputs.gamma
    )
    gradients = torch.autograd.grad(
        training_loss, tuple(named_parameters.values()), create_graph=True
    )
    adapted_parameters = virtual_adam_parameters(
        named_parameters,
        gradients,
        dqn_optimizer,
        outputs.learning_rate,
    )
    validation_parameters = dict(q_network.named_parameters())
    validation_parameters.update(adapted_parameters)
    validation_q_values = functional_call(
        q_network, validation_parameters, (validation_batch.observations,)
    )
    meta_loss = robust_outer_loss(
        validation_q_values, target_network, validation_batch, args, outputs
    )

    meta_optimizer.zero_grad()
    meta_loss.backward()
    grad_norm_tensor = nn.utils.clip_grad_norm_(
        controller.parameters(), args.meta_gradient_clip
    )
    if not torch.isfinite(grad_norm_tensor):
        raise FloatingPointError("non-finite confidence-controller gradient")
    meta_optimizer.step()
    detached_outputs = ControllerOutputs(
        *(value.detach() for value in outputs)
    )
    return (
        float(meta_loss.detach().item()),
        detached_outputs,
        float(grad_norm_tensor.item()),
    )


def ema(previous: float, value: float, alpha: float) -> float:
    return value if previous == 0.0 else (1.0 - alpha) * previous + alpha * value


def run_training(args: Args) -> dict:
    if args.meta_update_frequency % args.train_frequency != 0:
        raise ValueError(
            "meta_update_frequency must be divisible by train_frequency"
        )
    if args.env_id != "CartPole-v1":
        raise ValueError(
            "Partial-observation semantics are currently defined for CartPole-v1"
        )

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, args.capture_video, run_name)]
    )
    if not isinstance(envs.single_observation_space, gym.spaces.Box):
        raise TypeError("Only Box observation spaces are supported")
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("Only Discrete action spaces are supported")

    q_network = ConfidenceQNetwork(envs).to(device)
    target_network = ConfidenceQNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    dqn_optimizer = optim.Adam(
        q_network.main_parameters(), lr=args.learning_rate
    )
    controller = ConfidenceMetaController(args).to(device)
    meta_optimizer = optim.Adam(
        controller.parameters(), lr=args.meta_learning_rate
    )
    replay_buffer = ReplayBuffer(
        args.buffer_size, envs.single_observation_space, device
    )

    current_gamma = args.gamma
    current_learning_rate = args.learning_rate
    current_auxiliary_weight = args.initial_auxiliary_weight
    loss_ema = 0.0
    grad_ema = 0.0
    return_ema = 0.0
    episode_returns: list[float] = []
    controller_history: list[dict] = []
    start_time = time.time()
    observation, _ = envs.reset(seed=args.seed)
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
            progress = len(episode_returns) / max(1, args.total_episodes)
        else:
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            )
            progress = global_step / args.total_timesteps
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_full, _, _ = q_network(
                    torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    )
                )
            actions = q_full.argmax(dim=1).cpu().numpy()

        next_observation, rewards, terminations, truncations, infos = envs.step(
            actions
        )
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_return = float(
                        np.asarray(info["episode"]["r"]).item()
                    )
                    episode_returns.append(episode_return)
                    return_ema = ema(
                        return_ema,
                        episode_return,
                        args.telemetry_ema_alpha,
                    )
                    if writer is not None:
                        writer.add_scalar(
                            "charts/episodic_return",
                            episode_return,
                            global_step,
                        )
                    if (
                        args.total_episodes is not None
                        and len(episode_returns) >= args.total_episodes
                    ):
                        training_done = True
                        break

        real_next_observation = next_observation.copy()
        if "final_observation" in infos:
            for index, truncated in enumerate(truncations):
                if truncated:
                    real_next_observation[index] = infos[
                        "final_observation"
                    ][index]
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
            and len(replay_buffer) >= args.batch_size
        ):
            train_batch = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                state_q_values = q_network(train_batch.observations)
                controller_state = build_controller_state(
                    state_q_values,
                    loss_ema=loss_ema,
                    grad_ema=grad_ema,
                    return_ema=return_ema,
                    epsilon=epsilon,
                    progress=progress,
                    current_gamma=current_gamma,
                    current_learning_rate=current_learning_rate,
                    args=args,
                )

            if (
                args.meta_gradient
                and global_step % args.meta_update_frequency == 0
                and len(replay_buffer) >= args.meta_batch_size
            ):
                validation_batch = replay_buffer.sample(args.meta_batch_size)
                meta_loss, outputs, meta_grad_norm = meta_gradient_step(
                    q_network,
                    target_network,
                    dqn_optimizer,
                    controller,
                    meta_optimizer,
                    train_batch,
                    validation_batch,
                    controller_state,
                    args,
                )
                current_gamma = float(outputs.gamma.item())
                current_learning_rate = float(outputs.learning_rate.item())
                current_auxiliary_weight = float(outputs.auxiliary_weight.item())
                controller_history.append(
                    {
                        "step": global_step,
                        "gamma": current_gamma,
                        "learning_rate": current_learning_rate,
                        "auxiliary_weight": current_auxiliary_weight,
                        "meta_loss": meta_loss,
                        "meta_gradient_norm": meta_grad_norm,
                    }
                )
                if writer is not None:
                    writer.add_scalar("meta/loss", meta_loss, global_step)
                    writer.add_scalar(
                        "meta/gradient_norm", meta_grad_norm, global_step
                    )
            elif args.meta_gradient:
                with torch.no_grad():
                    outputs = controller(controller_state)
                current_gamma = float(outputs.gamma.item())
                current_learning_rate = float(outputs.learning_rate.item())
                current_auxiliary_weight = float(outputs.auxiliary_weight.item())

            dqn_optimizer.param_groups[0]["lr"] = current_learning_rate
            training_loss, _ = inner_loss(
                q_network(train_batch.observations),
                target_network,
                train_batch,
                current_gamma,
            )
            dqn_optimizer.zero_grad()
            training_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                q_network.main_parameters(), max_norm=10.0
            )
            dqn_optimizer.step()
            loss_ema = ema(
                loss_ema,
                float(training_loss.detach().item()),
                args.telemetry_ema_alpha,
            )
            grad_ema = ema(
                grad_ema,
                float(grad_norm.detach().item()),
                args.telemetry_ema_alpha,
            )

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar(
                    "losses/full", training_loss.item(), global_step
                )
                writer.add_scalar(
                    "meta/gamma", current_gamma, global_step
                )
                writer.add_scalar(
                    "meta/learning_rate",
                    current_learning_rate,
                    global_step,
                )
                writer.add_scalar(
                    "meta/auxiliary_weight",
                    current_auxiliary_weight,
                    global_step,
                )

        if global_step % args.target_network_frequency == 0:
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_parameter.copy_(
                        args.tau * parameter
                        + (1.0 - args.tau) * target_parameter
                    )

        if (
            global_step > 0
            and global_step % args.log_interval == 0
        ):
            sps = int(global_step / max(time.time() - start_time, 1e-9))
            last_return = episode_returns[-1] if episode_returns else 0.0
            print(
                f"step={global_step}, episodes={len(episode_returns)}, "
                f"return={last_return:.1f}, gamma={current_gamma:.5f}, "
                f"lr={current_learning_rate:.3e}, "
                f"aux_w={current_auxiliary_weight:.3f}, SPS={sps}",
                flush=True,
            )

    result = {
        "args": vars(args),
        "episode_returns": episode_returns,
        "controller_history": controller_history,
        "global_steps": global_step + 1,
        "total_episodes": args.total_episodes,
        "final_gamma": current_gamma,
        "final_learning_rate": current_learning_rate,
        "final_auxiliary_weight": current_auxiliary_weight,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_model:
        model_directory = Path("runs") / run_name
        model_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": q_network.state_dict(),
                "controller": controller.state_dict(),
                "result": result,
            },
            model_directory / f"{args.exp_name}.pt",
        )
    envs.close()
    if writer is not None:
        writer.close()
    return result


def main() -> None:
    run_training(tyro.cli(Args))


if __name__ == "__main__":
    main()
