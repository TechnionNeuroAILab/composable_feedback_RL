"""Per-channel local controllers: decentralized replacement for HyperController.

The centralized HyperController (``code/paper_hpo_dqn/hypercontroller.py``)
fits one coordinate-wise ridge model per hyperparameter from a single global
scalar feedback signal, jointly across all coordinates. That is a reasonable
design for tuning two or three DQN-wide knobs, but it does not scale
biologically: nothing in cortex/basal-ganglia loops has access to a global
scalar reward-change signal fed into a single joint regression.

Here each channel ``k`` gets its own small controller that only ever reads
statistics local to channel ``k`` (its own TD-error variance, its own value
function's fit quality, and pairwise comparisons against the *other*
channels that are directly observable where the channels interact --
routing/advantage covariance and policy-gradient conflict). Each controller
runs simple bounded proportional/EMA update rules -- no joint model, no
shared optimizer state -- and outputs the settings for its own channel only:
gate ``g_k``, routing-column scale (feeds ``B_{:,k}``), critic learning
rate, GAE ``lambda_k`` (and optionally ``gamma_k``), the shared-vs-specific
critic error weight ``alpha_k``, and its own sparsity/overlap pressure.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig


@dataclass(slots=True)
class ChannelStats:
    """Statistics local (or pairwise-local) to one channel, for one window."""

    td_variance: float
    usefulness: float
    redundancy: float
    gradient_conflict: float


@dataclass(slots=True)
class ChannelSettings:
    """Everything a local controller decides for the next rollout window."""

    gate: float
    column_scale: float
    critic_lr: float
    gae_lambda: float
    gamma: float
    alpha_specific: float
    sparsity_weight: float
    overlap_weight: float


def td_error_redundancy(td_errors: np.ndarray) -> np.ndarray:
    """Mean |corr(delta_k, delta_j)| over j != k. ``td_errors`` is ``(T, K)``."""

    num_channels = td_errors.shape[1]
    if num_channels == 1:
        return np.zeros(1, dtype=np.float64)
    corr = np.corrcoef(td_errors, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    redundancy = np.zeros(num_channels, dtype=np.float64)
    for k in range(num_channels):
        others = np.delete(np.abs(corr[k]), k)
        redundancy[k] = float(others.mean()) if others.size else 0.0
    return redundancy


def gradient_conflict_scores(channel_gradients: Sequence[Tensor]) -> np.ndarray:
    """Mean(-cos_sim) against the other channels' shared-trunk gradients.

    Positive => this channel's gradient tends to oppose the others (conflict).
    Negative => it tends to agree with them.
    """

    num_channels = len(channel_gradients)
    if num_channels == 1:
        return np.zeros(1, dtype=np.float64)
    flat = [g.flatten() for g in channel_gradients]
    norms = [g.norm().clamp_min(1e-8) for g in flat]
    cosine = np.zeros((num_channels, num_channels), dtype=np.float64)
    for i in range(num_channels):
        for j in range(num_channels):
            cosine[i, j] = float(torch.dot(flat[i], flat[j]) / (norms[i] * norms[j]))
    scores = np.zeros(num_channels, dtype=np.float64)
    for k in range(num_channels):
        others = np.delete(cosine[k], k)
        scores[k] = float(-others.mean()) if others.size else 0.0
    return scores


class LocalChannelController:
    """Bounded, rule-based adaptation for a single channel."""

    def __init__(
        self,
        channel_id: int,
        config: ExperimentConfig,
        initial_gamma: float,
        initial_lambda: float,
    ) -> None:
        self.id = channel_id
        self.config = config
        self.gate = 0.5
        self.column_scale = 1.0
        self.critic_lr_scale = 1.0
        self.gae_lambda = initial_lambda
        self.gamma = initial_gamma
        self.identity_gamma = initial_gamma
        self.alpha_specific = 0.5
        self.sparsity_weight = config.sparsity_coef
        self.overlap_weight = config.overlap_coef
        self._ema_variance: float | None = None
        self._ema_usefulness: float | None = None
        self._ema_redundancy: float | None = None
        self._ema_conflict: float | None = None
        self.history: list[dict[str, Any]] = []

    def _ema(self, previous: float | None, value: float) -> float:
        decay = self.config.controller_ema_decay
        return value if previous is None else decay * previous + (1.0 - decay) * value

    def observe_and_adapt(self, stats: ChannelStats) -> ChannelSettings:
        cfg = self.config
        self._ema_variance = self._ema(self._ema_variance, stats.td_variance)
        self._ema_usefulness = self._ema(self._ema_usefulness, stats.usefulness)
        self._ema_redundancy = self._ema(self._ema_redundancy, stats.redundancy)
        self._ema_conflict = self._ema(self._ema_conflict, stats.gradient_conflict)

        usefulness = float(np.clip(self._ema_usefulness, -1.0, 1.0))
        redundancy = float(np.clip(self._ema_redundancy, 0.0, 1.0))
        conflict = float(np.clip(self._ema_conflict, -1.0, 1.0))
        variance_pressure = math.tanh(
            self._ema_variance / max(cfg.target_td_variance, 1e-6) - 1.0
        )

        # Recruit this loop more when it is useful and not redundant/conflicting.
        recruit_signal = usefulness - 0.5 * redundancy - 0.5 * max(conflict, 0.0)
        self.gate = float(
            np.clip(self.gate + cfg.controller_gate_step * math.tanh(recruit_signal), 0.05, 1.0)
        )

        # Routing strength follows the same signal (sparse_b mode only).
        if cfg.routing_mode == "sparse_b":
            self.column_scale = float(
                np.clip(
                    self.column_scale
                    + cfg.controller_bscale_step * math.tanh(usefulness - max(conflict, 0.0)),
                    0.1,
                    3.0,
                )
            )

        # High own-TD-variance -> smaller critic step and lower lambda (less
        # variance-amplifying multi-step bootstrapping).
        self.critic_lr_scale = float(
            np.clip(
                self.critic_lr_scale * math.exp(-cfg.controller_lr_step * variance_pressure),
                cfg.critic_lr_scale_min,
                cfg.critic_lr_scale_max,
            )
        )
        self.gae_lambda = float(
            np.clip(
                self.gae_lambda - cfg.controller_lambda_step * variance_pressure,
                cfg.gae_lambda_min,
                cfg.gae_lambda_max,
            )
        )
        if cfg.adapt_gamma:
            # Usefulness alone has no cross-channel awareness: if every
            # channel's critic fits its own returns reasonably well, this
            # term pushes every channel toward gamma_max at once, erasing
            # the horizon spread the channels were built to have. The
            # reversion term pulls gamma_k back toward its own assigned
            # identity in proportion to how far it has drifted, keeping
            # channels distinguishable while still letting usefulness shift
            # each one by a bounded amount around its identity horizon.
            drift = cfg.controller_gamma_step * math.tanh(usefulness)
            reversion = cfg.gamma_reversion_rate * (self.gamma - self.identity_gamma)
            self.gamma = float(
                np.clip(self.gamma + drift - reversion, cfg.gamma_min, cfg.gamma_max)
            )

        # More useful and less redundant -> lean on this channel's own
        # target rather than the cross-channel consensus target.
        self.alpha_specific = float(
            np.clip(
                self.alpha_specific
                + cfg.controller_alpha_step * math.tanh(usefulness - redundancy),
                0.05,
                0.95,
            )
        )

        # Redundant/conflicting channels are pushed toward sparser routing (sparse_b only).
        if cfg.routing_mode == "sparse_b":
            self.sparsity_weight = float(cfg.sparsity_coef * (1.0 + 2.0 * redundancy))
            self.overlap_weight = float(
                cfg.overlap_coef * (1.0 + 2.0 * redundancy + max(conflict, 0.0))
            )
        else:
            self.sparsity_weight = 0.0
            self.overlap_weight = 0.0

        settings = ChannelSettings(
            gate=self.gate,
            column_scale=self.column_scale,
            critic_lr=self.critic_lr_scale * cfg.base_critic_lr,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            alpha_specific=self.alpha_specific,
            sparsity_weight=self.sparsity_weight,
            overlap_weight=self.overlap_weight,
        )
        self.history.append({"stats": asdict(stats), "settings": asdict(settings)})
        return settings


class ChannelControllerBank:
    """Owns one :class:`LocalChannelController` per channel."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        gammas = config.initial_gammas()
        lambdas = config.initial_lambdas()
        self.controllers = [
            LocalChannelController(k, config, gammas[k], lambdas[k])
            for k in range(config.num_channels)
        ]

    @property
    def gammas(self) -> list[float]:
        return [c.gamma for c in self.controllers]

    @property
    def lambdas(self) -> list[float]:
        return [c.gae_lambda for c in self.controllers]

    def build_stats(
        self,
        td_errors: np.ndarray,
        values: np.ndarray,
        returns: np.ndarray,
        channel_gradients: Sequence[Tensor],
    ) -> list[ChannelStats]:
        from .core import explained_variance

        variances = td_errors.var(axis=0)
        redundancy = td_error_redundancy(td_errors)
        conflict = gradient_conflict_scores(channel_gradients)
        num_channels = td_errors.shape[1]
        usefulness = np.array(
            [explained_variance(values[:, k], returns[:, k]) for k in range(num_channels)]
        )
        return [
            ChannelStats(
                td_variance=float(variances[k]),
                usefulness=float(usefulness[k]),
                redundancy=float(redundancy[k]),
                gradient_conflict=float(conflict[k]),
            )
            for k in range(num_channels)
        ]

    def adapt(self, stats: list[ChannelStats]) -> list[ChannelSettings]:
        return [
            controller.observe_and_adapt(stat)
            for controller, stat in zip(self.controllers, stats)
        ]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "gate": [c.gate for c in self.controllers],
            "column_scale": [c.column_scale for c in self.controllers],
            "critic_lr": [c.critic_lr_scale * self.config.base_critic_lr for c in self.controllers],
            "gae_lambda": [c.gae_lambda for c in self.controllers],
            "gamma": [c.gamma for c in self.controllers],
            "alpha_specific": [c.alpha_specific for c in self.controllers],
            "sparsity_weight": [c.sparsity_weight for c in self.controllers],
            "overlap_weight": [c.overlap_weight for c in self.controllers],
        }


__all__ = [
    "ChannelStats",
    "ChannelSettings",
    "LocalChannelController",
    "ChannelControllerBank",
    "td_error_redundancy",
    "gradient_conflict_scores",
]
