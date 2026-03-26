"""Training configuration for Bike DQN runs (multi-scenario / multi-seed friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    """Hyperparameters and paths for one training run."""

    scenario_name: str = "default"
    """Logical name for this scenario (used in default folder layout)."""

    n_lanes: int = 3
    seed: int = 1

    base_output_dir: Path = field(default_factory=lambda: Path("bike_outputs"))
    """Root directory; each run uses ``base_output_dir / scenario_name / seed_{seed}`` unless log_dir is set."""

    log_dir: Optional[Path] = None
    """If set, overrides the default path derived from scenario and seed."""

    total_timesteps: int = 10_000_000
    checkpoint_save_freq: int = 100_000
    final_model_name: str = "bike_model_final"
    checkpoint_name_prefix: str = "bike_model"

    # DQN (stable-baselines3)
    learning_rate: float = 1e-4
    exploration_fraction: float = 0.5
    exploration_final_eps: float = 0.05
    policy_net_arch: tuple[int, ...] = (256, 256)
    verbose: int = 0

    def resolved_log_dir(self) -> Path:
        if self.log_dir is not None:
            return Path(self.log_dir)
        return self.base_output_dir / self.scenario_name / f"seed_{self.seed}"

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "n_lanes": self.n_lanes,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "checkpoint_save_freq": self.checkpoint_save_freq,
            "learning_rate": self.learning_rate,
            "exploration_fraction": self.exploration_fraction,
            "exploration_final_eps": self.exploration_final_eps,
            "policy_net_arch": list(self.policy_net_arch),
            "log_dir": str(self.resolved_log_dir()),
        }
