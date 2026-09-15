"""Experiment configuration for the locally-controlled composable PPO agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


RoutingMode = Literal["sparse_b", "backprop"]


@dataclass(slots=True)
class ExperimentConfig:
    """Static shapes/schedule plus the *initial* per-channel values.

    Everything a channel's local controller is allowed to adapt (gate,
    routing scale, critic learning rate, GAE lambda/gamma, shared-vs-specific
    weight, sparsity/overlap pressure) starts here and is then mutated
    per-window by :class:`~ppo_local_channel_control.local_controller.LocalChannelController`.
    """

    env_id: str = "CartPole-v1"
    seed: int = 1
    device: str = "cpu"

    # Architecture: K_max critics / channels, M encoder-actor modules.
    # ``sparse_b`` (v3): learned routing matrix B + sparsity/overlap penalties.
    # ``backprop`` (v4): no B; each critic reads the full concatenated module
    # representation and learns its own projection via ordinary backprop.
    routing_mode: RoutingMode = "sparse_b"
    num_channels: int = 4
    num_modules: int = 4
    module_feature_dim: int = 32
    hidden_size: int = 64

    # PPO schedule.
    total_env_steps: int = 200_000
    rollout_steps: int = 2048
    minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    actor_lr: float = 3e-4
    base_critic_lr: float = 1e-3

    # Running observation/reward normalization (standard for MuJoCo-scale PPO;
    # without it value targets swing across wildly different scales and the
    # critics never settle). Episode returns are still logged in raw,
    # unnormalized units via gymnasium's RecordEpisodeStatistics wrapper,
    # which is placed before normalization in the wrapper stack.
    normalize_observations: bool = True
    normalize_rewards: bool = True
    observation_clip: float = 10.0
    reward_clip: float = 10.0
    reward_norm_gamma: float = 0.99

    # Bounds on the per-channel critic learning-rate multiplier the local
    # controller can apply to base_critic_lr. The lower bound matters: too
    # low and a channel's critic can get throttled to a crawl for the rest
    # of training once TD variance pressure pushes it down.
    critic_lr_scale_min: float = 0.3
    critic_lr_scale_max: float = 3.0

    # Per-channel GAE/discount bounds. Initial values are spread across the
    # range so channels start life differentiated (short- to long-horizon).
    gamma_min: float = 0.90
    gamma_max: float = 0.999
    gae_lambda_min: float = 0.70
    gae_lambda_max: float = 0.99

    # Base sparsity/overlap pressure; local controllers scale these per
    # channel based on observed redundancy.
    sparsity_coef: float = 1e-3
    overlap_coef: float = 1e-3

    # Local-controller step sizes and stat-tracking EMA decay.
    controller_ema_decay: float = 0.9
    controller_gate_step: float = 0.15
    controller_lr_step: float = 0.20
    controller_lambda_step: float = 0.05
    controller_gamma_step: float = 0.01
    controller_alpha_step: float = 0.15
    controller_bscale_step: float = 0.20
    target_td_variance: float = 1.0
    adapt_gamma: bool = True
    # Pulls gamma_k back toward the channel's assigned initial horizon each
    # window, scaled by how far it has drifted. Without this the usefulness-
    # driven step above has nothing to stop every channel converging onto
    # the same gamma (observed: all 4 channels locked to an identical gamma
    # within the first ~2.5% of a 5M-step HalfCheetah run). Set high enough
    # relative to controller_gamma_step that the steady-state drift stays
    # well inside the gap between two channels' assigned horizons.
    gamma_reversion_rate: float = 0.5

    # Hard bounds on the continuous-action policy's (state-independent)
    # log_std. An entropy bonus has no natural ceiling on a state-independent
    # std -- with nothing pulling it back, log_std drifts up for the entire
    # run (observed: policy entropy climbed monotonically for all 5M steps
    # instead of plateauing). log_std_max=0.5 caps std at ~1.65 on a
    # [-1, 1]-scaled action space.
    log_std_min: float = -3.0
    log_std_max: float = 0.5

    # Gradient-conflict diagnostic: evaluated once per window on a random
    # sub-batch (K backward passes through the shared trunk only).
    conflict_eval_batch: int = 128

    def __post_init__(self) -> None:
        if self.routing_mode not in ("sparse_b", "backprop"):
            raise ValueError("routing_mode must be 'sparse_b' or 'backprop'")
        if self.num_channels < 1:
            raise ValueError("num_channels (K_max) must be positive")
        if self.num_modules < 1:
            raise ValueError("num_modules must be positive")
        if self.rollout_steps < self.minibatches:
            raise ValueError("rollout_steps must be >= minibatches")
        if not 0.0 < self.gamma_min <= self.gamma_max < 1.0:
            raise ValueError("invalid gamma bounds")
        if not 0.0 < self.gae_lambda_min <= self.gae_lambda_max <= 1.0:
            raise ValueError("invalid gae_lambda bounds")

    def initial_gammas(self) -> list[float]:
        if self.num_channels == 1:
            return [self.gamma_max]
        return [
            self.gamma_min + (self.gamma_max - self.gamma_min) * i / (self.num_channels - 1)
            for i in range(self.num_channels)
        ]

    def initial_lambdas(self) -> list[float]:
        if self.num_channels == 1:
            return [self.gae_lambda_max]
        return [
            self.gae_lambda_min
            + (self.gae_lambda_max - self.gae_lambda_min) * i / (self.num_channels - 1)
            for i in range(self.num_channels)
        ]

    @property
    def minibatch_size(self) -> int:
        return max(1, self.rollout_steps // self.minibatches)

    @property
    def num_windows(self) -> int:
        return max(1, math.ceil(self.total_env_steps / self.rollout_steps))


def halfcheetah_v4_config(device: str = "cuda") -> ExperimentConfig:
    """HalfCheetah-v4 preset matching the v3 budget with backprop routing (no B)."""

    return ExperimentConfig(
        env_id="HalfCheetah-v4",
        device=device,
        routing_mode="backprop",
        num_channels=4,
        num_modules=4,
        module_feature_dim=32,
        hidden_size=64,
        total_env_steps=5_000_000,
        rollout_steps=2048,
        sparsity_coef=0.0,
        overlap_coef=0.0,
    )


def halfcheetah_v4_ablation_k1_config(device: str = "cuda") -> ExperimentConfig:
    """Fair v4 ablation: same backprop routing as v4 but K=1 (no multi-loop machinery)."""

    config = halfcheetah_v4_config(device=device)
    config.num_channels = 1
    return config


def smoke_config(env_id: str = "CartPole-v1", device: str = "cpu") -> ExperimentConfig:
    """Small, fast configuration for an integration smoke test."""

    return ExperimentConfig(
        env_id=env_id,
        device=device,
        num_channels=3,
        num_modules=3,
        module_feature_dim=8,
        hidden_size=16,
        total_env_steps=640,
        rollout_steps=64,
        minibatches=2,
        update_epochs=2,
        conflict_eval_batch=16,
    )


__all__ = [
    "ExperimentConfig",
    "RoutingMode",
    "halfcheetah_v4_config",
    "halfcheetah_v4_ablation_k1_config",
    "smoke_config",
]
