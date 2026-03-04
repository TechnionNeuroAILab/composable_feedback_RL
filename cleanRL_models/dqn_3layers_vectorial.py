# DQN with two linear layers and vectorial error (based on dqn_3layers.py)
#
# This script uses a fully linear Q-network (no activations) and implements
# vectorial error: per-feature reward r/num_inputs and feature contributions.
#
# 1) Extract weights between layers: w1 (input -> hidden), w2 (hidden -> output)
# 2) Compute contribution of each input/feature to the chosen action's Q-value
# 3) Backpropagate reward to the feature level as r/num_inputs (vectorial reward)
#
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

try:
    from cleanrl_utils.buffers import ReplayBuffer
except Exception:
    ReplayBuffer = None


class SimpleReplayBuffer:
    """Minimal replay buffer compatible with Gymnasium (e.g. Box with inf bounds)."""

    def __init__(self, buffer_size, obs_shape, n_actions, device):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        self.obs = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, actions, rewards, terminations, infos):
        obs = np.atleast_2d(obs)
        next_obs = np.atleast_2d(next_obs)
        actions = np.atleast_1d(actions)
        rewards = np.atleast_1d(rewards)
        terminations = np.atleast_1d(terminations)
        for i in range(len(rewards)):
            self.obs[self.ptr] = obs[i]
            self.next_obs[self.ptr] = next_obs[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr] = rewards[i]
            self.dones[self.ptr] = float(terminations[i])
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return type("Batch", (), {
            "observations": torch.tensor(self.obs[idx], device=self.device),
            "next_observations": torch.tensor(self.next_obs[idx], device=self.device),
            "actions": torch.tensor(self.actions[idx], device=self.device).unsqueeze(1),
            "rewards": torch.tensor(self.rewards[idx], device=self.device).unsqueeze(1),
            "dones": torch.tensor(self.dones[idx], device=self.device).unsqueeze(1),
        })()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    vectorial_loss_coef: float = 0.1
    """coefficient for per-feature vectorial loss L_i (0 = disable). L_i = MSE(contribution of feature i to chosen action, r/num_inputs)."""


def make_env(env_id, seed, idx, capture_video, run_name):
    """Build a single environment (used by SyncVectorEnv)."""
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


# -----------------------------------------------------------------------------
# Fully linear Q-network: input -> fc1 (w1) -> fc2 (w2) -> output (no ReLU)
# -----------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Fully linear Q-network with two layers. No nonlinearity between layers.
    Structure: obs -> fc1 -> fc2 -> Q-values.
    Layers are named so we can extract w1 and w2 for vectorial error.
    """

    def __init__(self, env):
        super().__init__()
        obs_size = np.array(env.single_observation_space.shape).prod()
        n_actions = env.single_action_space.n
        self.obs_size = obs_size
        self.n_actions = n_actions
        # w1: input -> hidden; w2: hidden -> output
        self.fc1 = nn.Linear(obs_size, 120)
        self.fc2 = nn.Linear(120, n_actions)

    def forward(self, x):
        """Linear forward: x -> fc1(x) -> fc2(.)."""
        x = self.fc1(x)
        return self.fc2(x)

    # -------------------------------------------------------------------------
    # 1) Extract weights between layers (w1 and w2)
    # -------------------------------------------------------------------------
    def get_w1_w2(self):
        """
        Extract weight matrices between layers.
        Returns:
            w1: (120, obs_size) - weights from input to hidden layer
            w2: (n_actions, 120) - weights from hidden to output layer
        """
        w1 = self.fc1.weight   # (out_hidden, obs_size)
        w2 = self.fc2.weight   # (n_actions, out_hidden)
        return w1, w2

    # -------------------------------------------------------------------------
    # 2) Contribution of each input/feature to the output (chosen action's Q)
    #    Same formula as 2layers_dqn_linear.ipynb: w_eff = w2 @ w1, contrib = w_eff * x
    # -------------------------------------------------------------------------
    def feature_contribution(self, observations, actions):
        """
        Per-feature contribution to Q(s,a) for the chosen action a.
        Uses the same formula as 2layers_dqn_linear.ipynb:
          w_eff = w2 @ w1   # (n_actions, n_features) effective weight input -> Q per action
          contribution[a, f] = w_eff[a, f] * x[f]   # additive amount feature f adds to Q(s,a)
        For a batch we return contributions for the chosen action only per sample.

        Args:
            observations: (batch, obs_size)
            actions: (batch,) or (batch, 1) - indices of chosen actions

        Returns:
            contribution: (batch, obs_size) - contribution of each feature to Q(s, a_chosen)
        """
        w1, w2 = self.get_w1_w2()
        # w_eff[a, f] = (dQ_a/dx_f) coefficient; (n_actions, obs_size)
        w_eff = w2 @ w1
        act = actions.flatten()  # (batch,)
        # Row i: w_eff[act[i], :] * observations[i, :] = contribution of each feature to Q(s_i, a_i)
        contributions = w_eff[act] * observations  # (batch, obs_size)
        return contributions

    # -------------------------------------------------------------------------
    # 3) Backpropagate reward to the feature level: r/num_inputs per feature
    # -------------------------------------------------------------------------
    @staticmethod
    def vectorial_reward(rewards, num_inputs, device):
        """
        Assign reward to each feature equally: each feature gets r/num_inputs.
        Used for vectorial error / composable feedback at the feature level.

        Args:
            rewards: (batch,) scalar reward per transition
            num_inputs: number of observation/input features
            device: torch device

        Returns:
            r_vec: (batch, num_inputs) with each row [r/n, r/n, ..., r/n]
        """
        batch = rewards.shape[0]
        r_per_feature = rewards.unsqueeze(1) / num_inputs  # (batch, 1)
        r_vec = r_per_feature.expand(batch, num_inputs).to(device)
        return r_vec


def per_feature_loss(contribution, r_vec):
    """
    Per-feature loss L_i: for each feature i, L_i = MSE(contribution of feature i to chosen action, r/num_inputs).
    So L_i is computed from the contribution of feature_i to action_k (chosen action), not from a global sum.

    Args:
        contribution: (batch, num_inputs) - contribution[b, i] = amount feature i adds to Q(s_b, a_b)
        r_vec: (batch, num_inputs) - r_vec[b, i] = r_b / num_inputs

    Returns:
        L_per_feature: (num_inputs,) - L_i for each feature i
        loss_vectorial: scalar - mean of L_per_feature (for backward)
    """
    num_inputs = contribution.shape[1]
    L_per_feature = torch.stack(
        [F.mse_loss(contribution[:, i], r_vec[:, i]) for i in range(num_inputs)]
    )
    loss_vectorial = L_per_feature.mean()
    return L_per_feature, loss_vectorial


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear decay from start_e to end_e over duration steps."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    num_inputs = q_network.obs_size

    try:
        if ReplayBuffer is not None:
            rb = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
            )
        else:
            raise NotImplementedError("use fallback")
    except Exception:
        obs_shape = tuple(envs.single_observation_space.shape)
        n_actions = int(envs.single_action_space.n)
        rb = SimpleReplayBuffer(args.buffer_size, obs_shape, n_actions, device)
    start_time = time.time()
    ep_returns = np.zeros(args.num_envs)
    ep_lengths = np.zeros(args.num_envs, dtype=int)

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        ep_returns += rewards
        ep_lengths += 1

        if "final_info" in infos:
            for idx, info in enumerate(infos["final_info"]):
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    ep_returns[idx] = 0
                    ep_lengths[idx] = 0
        for idx in range(envs.num_envs):
            if terminations[idx] or truncations[idx]:
                if ep_lengths[idx] > 0:
                    writer.add_scalar("charts/episodic_return", float(ep_returns[idx]), global_step)
                    writer.add_scalar("charts/episodic_length", int(ep_lengths[idx]), global_step)
                ep_returns[idx] = 0
                ep_lengths[idx] = 0

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                # --- Standard DQN TD loss ---
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss_td = F.mse_loss(td_target, old_val)

                # --- Per-feature loss L_i: contribution of feature_i to chosen action_k vs r/num_inputs ---
                contribution = q_network.feature_contribution(data.observations, data.actions)
                r_vec = QNetwork.vectorial_reward(data.rewards.flatten(), num_inputs, device)
                L_per_feature, loss_vectorial = per_feature_loss(contribution, r_vec)
                loss = loss_td + args.vectorial_loss_coef * loss_vectorial

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss_td, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("losses/vectorial_loss", loss_vectorial, global_step)
                    writer.add_scalar("vectorial/reward_per_feature_mean", r_vec.mean().item(), global_step)
                    writer.add_scalar("vectorial/contribution_norm_mean", contribution.norm(dim=1).mean().item(), global_step)
                    for i in range(min(num_inputs, 4)):  # log first 4 features (e.g. CartPole)
                        writer.add_scalar(f"losses/L_i_feature_{i}", L_per_feature[i], global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate
        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
