"""Networks for the composable PPO agent: shared encoder/actor, K_max critics,
sparse module routing, and soft per-channel activation gates.

Design (see README.md for the full write-up and the mapping onto the
routing/feedback-matrix formalism in ``paper/feedback_matrix_models.tex``):

- ``ModularEncoder`` splits the shared trunk into ``M`` parallel modules, each
  producing a feature block ``z_m``. Both the actor and the critics are read
  off these blocks, so nothing downstream is architecturally private to one
  channel.
- ``ChannelRouter`` owns the sparse routing matrix ``B in R^{M x K}`` (one
  column per channel, softplus + column-normalized so each channel's routing
  weights sum to one across modules) and the per-channel gates ``g_k in
  [0, 1]``. ``B`` is learned by backprop through the critic/actor losses;
  ``g`` and a per-channel routing-column scale are set externally, once per
  rollout window, by each channel's :class:`~ppo_local_channel_control.local_controller.LocalChannelController`.
- The actor reads a gate-scaled combination of module blocks
  (``module_scale_m = sum_k B_mk * g_k``), so a channel that is not recruited
  (``g_k -> 0``) stops contributing to the shared representation. Each critic
  ``k`` reads its own routed combination ``c_k = sum_m B_mk * z_m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


def seed_everything(seed: int) -> np.random.Generator:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


class ModularEncoder(nn.Module):
    """``M`` independent small MLPs mapping observations to feature blocks."""

    def __init__(self, obs_dim: int, num_modules: int, feature_dim: int, hidden_size: int) -> None:
        super().__init__()
        if obs_dim <= 0 or num_modules <= 0 or feature_dim <= 0:
            raise ValueError("obs_dim, num_modules, and feature_dim must be positive")
        self.num_modules = num_modules
        self.feature_dim = feature_dim
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, feature_dim),
                    nn.Tanh(),
                )
                for _ in range(num_modules)
            ]
        )

    def forward(self, observations: Tensor) -> Tensor:
        """Return module feature blocks, shape ``(batch, M, feature_dim)``."""

        return torch.stack([branch(observations) for branch in self.branches], dim=1)


class ChannelGates(nn.Module):
    """Per-channel recruitment gates only (v4 backprop mode: no routing matrix B)."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer("gate", torch.full((num_channels,), 0.5))

    def set_gates(self, values: Sequence[float]) -> None:
        self.gate.copy_(torch.as_tensor(values, dtype=self.gate.dtype, device=self.gate.device))


class ChannelRouter(nn.Module):
    """Sparse routing matrix ``B`` and per-channel gates ``g``.

    ``raw_B`` is a learned parameter; ``gate`` and ``column_scale`` are
    controller-set buffers (no gradient), read fresh at the start of each
    rollout window.
    """

    def __init__(self, num_modules: int, num_channels: int) -> None:
        super().__init__()
        self.num_modules = num_modules
        self.num_channels = num_channels
        self.raw_B = nn.Parameter(torch.zeros(num_modules, num_channels))
        self.register_buffer("gate", torch.full((num_channels,), 0.5))
        self.register_buffer("column_scale", torch.ones(num_channels))

    def set_gates(self, values: Sequence[float]) -> None:
        self.gate.copy_(torch.as_tensor(values, dtype=self.gate.dtype, device=self.gate.device))

    def set_column_scales(self, values: Sequence[float]) -> None:
        self.column_scale.copy_(
            torch.as_tensor(values, dtype=self.column_scale.dtype, device=self.column_scale.device)
        )

    def routing_matrix(self) -> Tensor:
        """Return ``B``: non-negative, each column normalized to sum to one."""

        weights = nn.functional.softplus(self.raw_B) * self.column_scale.unsqueeze(0)
        column_sums = weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return weights / column_sums

    def module_recruitment_scale(self, routing: Tensor) -> Tensor:
        """Per-module scale ``sum_k B_mk * g_k``, shape ``(M,)``."""

        return (routing * self.gate.unsqueeze(0)).sum(dim=1)

    def column_entropy(self, routing: Tensor) -> Tensor:
        """Per-channel entropy of its routing column (low = sparse/concentrated)."""

        safe = routing.clamp_min(1e-8)
        return -(safe * safe.log()).sum(dim=0)

    def pairwise_column_similarity(self, routing: Tensor) -> Tensor:
        """``(K, K)`` cosine similarity between routing columns."""

        normalized = nn.functional.normalize(routing, dim=0, eps=1e-8)
        return normalized.t() @ normalized


class SharedActor(nn.Module):
    """One shared policy head reading the gate-scaled module representation."""

    def __init__(
        self,
        num_modules: int,
        feature_dim: int,
        hidden_size: int,
        action_dim: int,
        discrete: bool,
        log_std_min: float = -3.0,
        log_std_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.discrete = discrete
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.body = nn.Sequential(
            nn.Linear(num_modules * feature_dim, hidden_size),
            nn.Tanh(),
        )
        if discrete:
            self.head = nn.Linear(hidden_size, action_dim)
        else:
            self.mean = nn.Linear(hidden_size, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, actor_input: Tensor) -> Categorical | Normal:
        hidden = self.body(actor_input)
        if self.discrete:
            return Categorical(logits=self.head(hidden))
        mean = self.mean(hidden)
        # Entropy bonuses push log_std toward +inf with nothing to pull it
        # back for a state-independent std; clamp so exploration can't run
        # away and dominate the loss over a multi-million-step run.
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ChannelCritic(nn.Module):
    """One value head ``V_k`` for a single channel's routed features."""

    def __init__(self, feature_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, routed_features: Tensor) -> Tensor:
        return self.net(routed_features).squeeze(-1)


@dataclass(slots=True)
class ForwardResult:
    distribution: Categorical | Normal
    values: Tensor  # (batch, K)
    routing: Tensor | None  # (M, K) when sparse_b routing is active


class ComposablePPOAgent(nn.Module):
    """Shared encoder/actor with K_max value critics and optional sparse routing."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        discrete: bool,
        num_channels: int,
        num_modules: int,
        feature_dim: int,
        hidden_size: int,
        log_std_min: float = -3.0,
        log_std_max: float = 0.5,
        routing_mode: str = "sparse_b",
    ) -> None:
        super().__init__()
        if routing_mode not in ("sparse_b", "backprop"):
            raise ValueError("routing_mode must be 'sparse_b' or 'backprop'")
        self.routing_mode = routing_mode
        self.K = num_channels
        self.M = num_modules
        self.discrete = discrete
        self.encoder = ModularEncoder(obs_dim, num_modules, feature_dim, hidden_size)
        if routing_mode == "sparse_b":
            self.router: ChannelRouter | ChannelGates = ChannelRouter(num_modules, num_channels)
        else:
            self.router = ChannelGates(num_channels)
        self.actor = SharedActor(
            num_modules, feature_dim, hidden_size, action_dim, discrete, log_std_min, log_std_max
        )
        critic_input_dim = num_modules * feature_dim if routing_mode == "backprop" else feature_dim
        self.critics = nn.ModuleList(
            [ChannelCritic(critic_input_dim, hidden_size) for _ in range(num_channels)]
        )

    @property
    def uses_sparse_routing(self) -> bool:
        return self.routing_mode == "sparse_b"

    def encode(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        blocks = self.encoder(observations)  # (batch, M, F)
        if self.routing_mode == "sparse_b":
            router = self.router
            assert isinstance(router, ChannelRouter)
            routing = router.routing_matrix()  # (M, K)
            module_scale = router.module_recruitment_scale(routing)  # (M,)
            actor_input = (blocks * module_scale.view(1, -1, 1)).flatten(start_dim=1)
            critic_inputs = torch.einsum("bmf,mk->bkf", blocks, routing)  # (batch, K, F)
            return actor_input, critic_inputs, routing

        flat = blocks.flatten(start_dim=1)  # (batch, M*F)
        critic_inputs = flat.unsqueeze(1).expand(-1, self.K, -1)
        return flat, critic_inputs, None

    def forward(self, observations: Tensor) -> ForwardResult:
        actor_input, critic_inputs, routing = self.encode(observations)
        dist = self.actor.distribution(actor_input)
        values = torch.stack(
            [critic(critic_inputs[:, k]) for k, critic in enumerate(self.critics)], dim=1
        )
        return ForwardResult(dist, values, routing)

    def action_log_prob(self, distribution: Categorical | Normal, action: Tensor) -> Tensor:
        log_prob = distribution.log_prob(action)
        return log_prob if self.discrete else log_prob.sum(-1)

    @torch.no_grad()
    def act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = self.forward(observations)
        action = result.distribution.sample()
        log_prob = self.action_log_prob(result.distribution, action)
        return action, log_prob, result.values

    def shared_trunk_parameters(self) -> list[nn.Parameter]:
        """Parameters common to every channel's actor-path gradient (for conflict diagnostics)."""

        return list(self.encoder.parameters())


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: np.ndarray,
    gammas: Sequence[float],
    lambdas: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel GAE. Shapes: rewards/dones (T,), values (T, K), bootstrap_value (K,).

    Returns ``(advantages, returns)``, each ``(T, K)``.
    """

    steps, num_channels = values.shape
    advantages = np.zeros((steps, num_channels), dtype=np.float64)
    last_gae = np.zeros(num_channels, dtype=np.float64)
    for t in reversed(range(steps)):
        not_done = 1.0 - dones[t]
        next_value = bootstrap_value if t == steps - 1 else values[t + 1]
        for k in range(num_channels):
            delta = rewards[t] + gammas[k] * next_value[k] * not_done - values[t, k]
            last_gae[k] = delta + gammas[k] * lambdas[k] * not_done * last_gae[k]
            advantages[t, k] = last_gae[k]
    returns = advantages + values
    return advantages, returns


def compute_td_errors(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: np.ndarray,
    gammas: Sequence[float],
) -> np.ndarray:
    """One-step TD error ``delta_k,t = r_t + gamma_k V_k(s_{t+1}) (1-done) - V_k(s_t)``."""

    steps, num_channels = values.shape
    next_values = np.vstack([values[1:], bootstrap_value[None, :]])
    not_done = (1.0 - dones)[:, None]
    gamma_arr = np.asarray(gammas, dtype=np.float64)[None, :]
    return rewards[:, None] + gamma_arr * next_values * not_done - values


def explained_variance(predicted: np.ndarray, target: np.ndarray) -> float:
    target_var = np.var(target)
    if target_var < 1e-8:
        return 0.0
    return float(1.0 - np.var(target - predicted) / target_var)


__all__ = [
    "seed_everything",
    "ModularEncoder",
    "ChannelGates",
    "ChannelRouter",
    "SharedActor",
    "ChannelCritic",
    "ForwardResult",
    "ComposablePPOAgent",
    "compute_gae",
    "compute_td_errors",
    "explained_variance",
]
