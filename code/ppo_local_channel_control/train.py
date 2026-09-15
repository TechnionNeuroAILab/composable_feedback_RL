"""Training loop for the composable PPO agent with per-channel local control.

Direct invocation works without relying on ``code`` being an importable
package (mirrors ``code/paper_hpo_dqn/run_experiment.py``):

    python code/ppo_local_channel_control/train.py --smoke
    python code/ppo_local_channel_control/train.py \
        --env-id CartPole-v1 --total-env-steps 200000 --output-dir results/ppo_local

Each rollout window: collect ``rollout_steps`` transitions with the agent's
current gates/routing, run PPO epochs (shared actor loss on a gate-weighted
combined advantage, per-channel critic loss blended by ``alpha_k`` between
its own and the cross-channel consensus target, and sparsity/overlap
penalties on the routing matrix). Afterwards each channel's local
controller observes its own window statistics (TD variance, usefulness,
redundancy, gradient conflict) and the settings it returns are applied for
the next window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ppo_local_channel_control.config import (
        ExperimentConfig,
        halfcheetah_v4_ablation_k1_config,
        halfcheetah_v4_config,
        smoke_config,
    )
    from ppo_local_channel_control.core import (
        ComposablePPOAgent,
        compute_gae,
        compute_td_errors,
        seed_everything,
    )
    from ppo_local_channel_control.local_controller import ChannelControllerBank
else:
    from .config import (
        ExperimentConfig,
        halfcheetah_v4_ablation_k1_config,
        halfcheetah_v4_config,
        smoke_config,
    )
    from .core import ComposablePPOAgent, compute_gae, compute_td_errors, seed_everything
    from .local_controller import ChannelControllerBank

import gymnasium as gym


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RolloutBuffer:
    def __init__(
        self, capacity: int, obs_dim: int, action_dim: int, num_channels: int, discrete: bool
    ) -> None:
        self.capacity = capacity
        self.discrete = discrete
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = (
            np.zeros(capacity, dtype=np.int64)
            if discrete
            else np.zeros((capacity, action_dim), dtype=np.float32)
        )
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros((capacity, num_channels), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, i: int, obs, action, log_prob, values, reward, done) -> None:
        self.observations[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = values
        self.rewards[i] = reward
        self.dones[i] = done


def make_env(config: ExperimentConfig) -> gym.Env:
    """Build the training env with raw-return logging and running normalization.

    Order matters: ``RecordEpisodeStatistics`` sits directly on the base env
    so it captures true, unnormalized episode returns (available in
    ``info["episode"]["r"]`` when an episode ends) before any outer wrapper
    rescales observations/rewards. Continuous action spaces get their
    actions clipped to the box before reaching the simulator, matching
    standard PPO-on-MuJoCo practice.
    """

    env = gym.make(config.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    if config.normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -config.observation_clip, config.observation_clip),
            env.observation_space,
        )
    if config.normalize_rewards:
        env = gym.wrappers.NormalizeReward(env, gamma=config.reward_norm_gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -config.reward_clip, config.reward_clip)
        )
    return env


def build_agent(env: gym.Env, config: ExperimentConfig) -> ComposablePPOAgent:
    obs_dim = int(np.prod(env.observation_space.shape))
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = int(env.action_space.n) if discrete else int(env.action_space.shape[0])
    return ComposablePPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete=discrete,
        num_channels=config.num_channels,
        num_modules=config.num_modules,
        feature_dim=config.module_feature_dim,
        hidden_size=config.hidden_size,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        routing_mode=config.routing_mode,
    ).to(config.device)


def make_optimizer(agent: ComposablePPOAgent, config: ExperimentConfig) -> torch.optim.Adam:
    router_params = list(agent.router.parameters()) if agent.uses_sparse_routing else []
    shared_params = list(agent.encoder.parameters()) + router_params + list(agent.actor.parameters())
    groups = [{"params": shared_params, "lr": config.actor_lr, "name": "shared"}]
    for k, critic in enumerate(agent.critics):
        groups.append(
            {"params": list(critic.parameters()), "lr": config.base_critic_lr, "name": f"critic_{k}"}
        )
    return torch.optim.Adam(groups)


def set_critic_learning_rates(optimizer: torch.optim.Adam, critic_lrs: list[float]) -> None:
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name.startswith("critic_"):
            group["lr"] = critic_lrs[int(name.split("_")[1])]


def collect_rollout(
    env: gym.Env,
    agent: ComposablePPOAgent,
    buffer: RolloutBuffer,
    obs: np.ndarray,
    config: ExperimentConfig,
    episode_returns: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    device = config.device
    for i in range(buffer.capacity):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, log_prob, values = agent.act(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy()
        env_action = int(action_np) if agent.discrete else action_np
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated  # time-limit truncation bootstraps; true termination does not
        buffer.add(
            i,
            obs,
            action_np,
            float(log_prob.item()),
            values.squeeze(0).cpu().numpy(),
            float(reward),
            float(done),
        )
        obs = next_obs
        if terminated or truncated:
            episode_info = info.get("episode")
            if episode_info is not None:
                episode_returns.append(float(episode_info["r"]))
            obs, _ = env.reset()
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        bootstrap_value = agent.forward(obs_tensor).values.squeeze(0).cpu().numpy()
    return obs, bootstrap_value


def compute_channel_gradients(
    agent: ComposablePPOAgent, obs: Tensor, actions: Tensor, advantages: np.ndarray
) -> list[Tensor]:
    """One fresh forward+backward per channel through the shared encoder only."""

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=obs.device)
    trunk_params = agent.shared_trunk_parameters()
    gradients = []
    for k in range(agent.K):
        agent.zero_grad(set_to_none=True)
        result = agent.forward(obs)
        log_prob = agent.action_log_prob(result.distribution, actions)
        loss_k = -(log_prob * advantages_t[:, k]).mean()
        grads = torch.autograd.grad(loss_k, trunk_params, allow_unused=True, retain_graph=False)
        flat = torch.cat(
            [
                g.flatten() if g is not None else torch.zeros_like(p).flatten()
                for g, p in zip(grads, trunk_params)
            ]
        )
        gradients.append(flat.detach())
    agent.zero_grad(set_to_none=True)
    return gradients


def ppo_update(
    agent: ComposablePPOAgent,
    optimizer: torch.optim.Adam,
    buffer: RolloutBuffer,
    returns: np.ndarray,
    combined_advantage: np.ndarray,
    shared_returns: np.ndarray,
    alpha_specific: list[float],
    sparsity_weights: list[float],
    overlap_weights: list[float],
    config: ExperimentConfig,
) -> dict[str, float]:
    device = config.device
    obs = torch.as_tensor(buffer.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(
        buffer.actions, dtype=torch.long if agent.discrete else torch.float32, device=device
    )
    old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=device)
    combined_advantage_t = torch.as_tensor(combined_advantage, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    shared_returns_t = torch.as_tensor(shared_returns, dtype=torch.float32, device=device)
    alpha_t = torch.as_tensor(alpha_specific, dtype=torch.float32, device=device)
    sparsity_t = torch.as_tensor(sparsity_weights, dtype=torch.float32, device=device)
    overlap_t = torch.as_tensor(overlap_weights, dtype=torch.float32, device=device)

    num_samples = buffer.capacity
    indices = np.arange(num_samples)
    minibatch_size = config.minibatch_size
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "sparsity_loss": 0.0,
        "overlap_loss": 0.0,
        "grad_norm": 0.0,
        "clip_fraction": 0.0,
    }
    num_updates = 0
    K = agent.K
    eye = torch.eye(K, device=device)
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, minibatch_size):
            batch_idx = torch.as_tensor(
                indices[start : start + minibatch_size], device=device, dtype=torch.long
            )
            result = agent.forward(obs[batch_idx])
            new_log_prob = agent.action_log_prob(result.distribution, actions[batch_idx])
            ratio = (new_log_prob - old_log_probs[batch_idx]).exp()
            adv_batch = combined_advantage_t[batch_idx]
            unclipped = ratio * adv_batch
            clipped = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * adv_batch
            policy_loss = -torch.min(unclipped, clipped).mean()

            entropy = result.distribution.entropy()
            if entropy.ndim > 1:
                entropy = entropy.sum(-1)
            entropy = entropy.mean()

            values_batch = result.values  # (batch, K)
            returns_batch = returns_t[batch_idx]
            shared_returns_batch = shared_returns_t[batch_idx].unsqueeze(-1).expand_as(values_batch)
            specific_loss = (values_batch - returns_batch).pow(2)
            shared_loss = (values_batch - shared_returns_batch).pow(2)
            per_channel_value_loss = (
                alpha_t.unsqueeze(0) * specific_loss + (1 - alpha_t).unsqueeze(0) * shared_loss
            ).mean(dim=0)
            value_loss = per_channel_value_loss.sum()

            if agent.uses_sparse_routing and result.routing is not None:
                router = agent.router
                assert hasattr(router, "column_entropy")
                column_entropy = router.column_entropy(result.routing)  # (K,)
                sparsity_loss = (sparsity_t * column_entropy).sum()
                similarity = router.pairwise_column_similarity(result.routing)  # (K, K)
                off_diag_sq = (similarity**2) * (1 - eye)
                overlap_per_channel = off_diag_sq.sum(dim=1) / max(K - 1, 1)
                overlap_loss = (overlap_t * overlap_per_channel).sum()
            else:
                sparsity_loss = torch.zeros((), device=device)
                overlap_loss = torch.zeros((), device=device)

            total_loss = (
                policy_loss
                - config.entropy_coef * entropy
                + config.value_coef * value_loss
                + sparsity_loss
                + overlap_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy.detach())
            metrics["sparsity_loss"] += float(sparsity_loss.detach())
            metrics["overlap_loss"] += float(overlap_loss.detach())
            metrics["grad_norm"] += float(grad_norm)
            metrics["clip_fraction"] += float(clip_fraction)
            num_updates += 1
    for key in metrics:
        metrics[key] /= max(num_updates, 1)
    return metrics


def run(config: ExperimentConfig) -> dict[str, Any]:
    rng = seed_everything(config.seed)
    env = make_env(config)
    obs, _ = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    agent = build_agent(env, config)
    optimizer = make_optimizer(agent, config)
    controller_bank = ChannelControllerBank(config)
    agent.router.set_gates([c.gate for c in controller_bank.controllers])
    if agent.uses_sparse_routing and hasattr(agent.router, "set_column_scales"):
        agent.router.set_column_scales([c.column_scale for c in controller_bank.controllers])

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n) if agent.discrete else int(env.action_space.shape[0])

    episode_returns: list[float] = []
    window_logs: list[dict[str, Any]] = []
    total_steps = 0
    window = 0

    while total_steps < config.total_env_steps:
        window += 1
        buffer = RolloutBuffer(
            config.rollout_steps, obs_dim, action_dim, config.num_channels, agent.discrete
        )
        obs, bootstrap_value = collect_rollout(env, agent, buffer, obs, config, episode_returns)
        total_steps += buffer.capacity

        gammas = controller_bank.gammas
        lambdas = controller_bank.lambdas
        rewards64 = buffer.rewards.astype(np.float64)
        values64 = buffer.values.astype(np.float64)
        dones64 = buffer.dones.astype(np.float64)
        advantages, returns = compute_gae(
            rewards64, values64, dones64, bootstrap_value.astype(np.float64), gammas, lambdas
        )
        td_errors = compute_td_errors(rewards64, values64, dones64, bootstrap_value.astype(np.float64), gammas)

        gate = np.asarray([c.gate for c in controller_bank.controllers], dtype=np.float64)
        gate_sum = max(float(gate.sum()), 1e-6)
        combined_advantage = (advantages * gate[None, :]).sum(axis=1) / gate_sum
        combined_advantage = (combined_advantage - combined_advantage.mean()) / (
            combined_advantage.std() + 1e-8
        )
        shared_returns = (returns * gate[None, :]).sum(axis=1) / gate_sum

        alpha = [c.alpha_specific for c in controller_bank.controllers]
        sparsity_weights = [c.sparsity_weight for c in controller_bank.controllers]
        overlap_weights = [c.overlap_weight for c in controller_bank.controllers]
        critic_lrs = [c.critic_lr_scale * config.base_critic_lr for c in controller_bank.controllers]
        set_critic_learning_rates(optimizer, critic_lrs)

        update_stats = ppo_update(
            agent,
            optimizer,
            buffer,
            returns,
            combined_advantage,
            shared_returns,
            alpha,
            sparsity_weights,
            overlap_weights,
            config,
        )

        eval_size = min(config.conflict_eval_batch, buffer.capacity)
        eval_idx = rng.choice(buffer.capacity, size=eval_size, replace=False)
        obs_sub = torch.as_tensor(buffer.observations[eval_idx], dtype=torch.float32, device=config.device)
        actions_sub = torch.as_tensor(
            buffer.actions[eval_idx],
            dtype=torch.long if agent.discrete else torch.float32,
            device=config.device,
        )
        channel_gradients = compute_channel_gradients(agent, obs_sub, actions_sub, advantages[eval_idx])

        stats = controller_bank.build_stats(td_errors, values64, returns, channel_gradients)
        controller_bank.adapt(stats)

        agent.router.set_gates([c.gate for c in controller_bank.controllers])
        if agent.uses_sparse_routing and hasattr(agent.router, "set_column_scales"):
            agent.router.set_column_scales([c.column_scale for c in controller_bank.controllers])

        mean_return = float(np.mean(episode_returns[-10:])) if episode_returns else float("nan")
        log_entry = {
            "window": window,
            "total_steps": total_steps,
            "mean_recent_return": mean_return,
            "episodes": len(episode_returns),
            **update_stats,
            "controllers": controller_bank.diagnostics(),
        }
        window_logs.append(log_entry)
        print(
            f"[window {window}] steps={total_steps} mean_return={mean_return:.2f} "
            f"policy_loss={update_stats['policy_loss']:.4f} value_loss={update_stats['value_loss']:.4f} "
            f"gates={[round(g, 2) for g in gate.tolist()]}"
        )

    env.close()
    return {"episode_returns": episode_returns, "windows": window_logs}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--num-modules", type=int, default=None)
    parser.add_argument(
        "--routing-mode",
        choices=("sparse_b", "backprop"),
        default=None,
        help="sparse_b (v3, learned B) or backprop (v4, no B matrix)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Small fast config for an integration smoke test")
    parser.add_argument(
        "--halfcheetah-v4",
        action="store_true",
        help="HalfCheetah-v4 preset: 5M steps, K=4, backprop routing (no B)",
    )
    parser.add_argument(
        "--halfcheetah-v4-ablation-k1",
        action="store_true",
        help="Fair v4 ablation: 5M steps, K=1, backprop routing (no B)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.smoke:
        config = smoke_config(env_id=args.env_id, device=args.device)
    elif args.halfcheetah_v4_ablation_k1:
        config = halfcheetah_v4_ablation_k1_config(device=args.device)
    elif args.halfcheetah_v4:
        config = halfcheetah_v4_config(device=args.device)
    else:
        config = ExperimentConfig(env_id=args.env_id, device=args.device)
    config.seed = args.seed
    if args.total_env_steps is not None:
        config.total_env_steps = args.total_env_steps
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    if args.num_channels is not None:
        config.num_channels = args.num_channels
    if args.num_modules is not None:
        config.num_modules = args.num_modules
    if args.routing_mode is not None:
        config.routing_mode = args.routing_mode
    config.__post_init__()

    start = time.time()
    result = run(config)
    duration = time.time() - start
    print(f"Finished {config.total_env_steps} env steps in {duration:.1f}s; episodes={len(result['episode_returns'])}")

    if args.output_dir:
        out = Path(args.output_dir)
        write_json(out / "config.json", asdict(config))
        write_json(out / "windows.json", result["windows"])
        write_json(out / "episode_returns.json", result["episode_returns"])
        print(f"Wrote outputs to {out}")
    return result


if __name__ == "__main__":
    main()
