# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
#
# [ConfGate, single agent] cleanrl_dqn_bike.py plus aux heads (two by default, --heads per_obs: one per observation entry) that scale the learning rate -- nothing else changes
# (one agent, its own replay buffer, the same epsilon schedule; no population, mutation or selection).
#   * two linear heads on the Q-network's last hidden layer (84 units), each fed an observation group with the rest
#     zeroed (CartPole: cart = obs[0:2], pole = obs[2:4]; HalfCheetah: pose = obs[0:8], velocity = obs[8:17]).
#     Trunk detached: the heads never act and never change the Q-network's gradient. They are fitted on the same
#     training batch to the same TD target, with their own Adam optimizer (one step per Q update).
#   * m = mean pairwise agreement of the heads' greedy actions over the batch (per joint on HalfCheetah);
#     for two heads this is the share of states where they pick the same action.
#   * lr used for this Q update = learning_rate * (lr_floor + lr_slope * m). No evolution here, so floor and slope are
#     fixed (default 0.5 / 0.5, the initial values of the SEARL version). --lr-floor 1 --lr-slope 0 = plain DQN.
#   * --mix-lambda L > 0 (ConfGate-mix): the heads also take part in ACTING: Q_mix = (1-L) Q_full + L mean_k Q_k, used when
#     collecting data (epsilon-greedy) and in the greedy bike evaluation. The TD target and the Q update are unchanged.
#     No evolution here, so L is fixed. --mix-lambda 0 (default) = heads never act.
# Head init uses a private torch generator and the heads reuse the Q batch, so with floor 1 / slope 0 the run is
# identical to cleanrl_dqn_bike.py for the same seed.
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

from cleanrl_utils.buffers import ReplayBuffer


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
    env_id: str = "bike_v2.1"
    """the id of the environment"""
    total_timesteps: int = 480000
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
    csv_path: str = ""
    """[added for this reproduction] if set, append (global_step, episodic_return) rows to this CSV"""
    lanes: int = 5
    """[bike grid] number of lanes of bike_v2.1"""
    solutions: int = 8
    """[bike grid] number of solutions (fall-free routes) of bike_v2.1"""
    eval_every: int = 0
    """[bike grid] if > 0: every eval_every steps, run a greedy (no exploration) evaluation on fixed fresh boards"""
    eval_episodes: int = 10
    eval_seed0: int = 5000
    eval_csv: str = ""
    """[bike grid] where to write global_step, greedy_return"""
    lr_floor: float = 0.5
    """[ConfGate] lr multiplier at m = 0"""
    lr_slope: float = 0.5
    """[ConfGate] lr multiplier gained per unit of m (lr scale = lr_floor + lr_slope * m)"""
    mix_lambda: float = 0.0
    """[ConfGate-mix] weight of the heads in the acting policy: Q_mix = (1 - lam) * Q_full + lam * mean_k Q_k(group-k view).
    0 = heads never act (default). Used when collecting data and in the greedy bike evaluation; the TD target is untouched."""
    heads: str = "groups"
    """[ConfGate] groups: two heads on observation groups; per_obs: one head per observation entry"""
    conf_log_path: str = ""
    """[ConfGate] if set, write global_step, m, lr_eff of every Q update to this CSV"""


BIKE_KW = {"lanes": 5, "solutions": 8}  # [bike grid] set from the CLI before the envs are built


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # [changed for this reproduction] bike v2.1 instead of gym.make(env_id); common_bike adds
        # RecordEpisodeStatistics (raw return) and the shared reward/20 scaling
        if env_id == "bike_v2.1":
            from common_bike import make_bike_env
            env = make_bike_env(seed * 100 + idx, lanes=BIKE_KW["lanes"], solutions=BIKE_KW["solutions"])
        else:   # any gymnasium discrete env (e.g. CartPole-v1): CleanRL's own env construction
            try:
                import ale_py
                gym.register_envs(ale_py)
            except Exception:
                pass
            from discrete_halfcheetah import prepare_env
            env = gym.wrappers.RecordEpisodeStatistics(prepare_env(gym.make(env_id), env_id))
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, _n_outputs(env.single_action_space)),
        )

    def forward(self, x):
        return self.network(x)


def _n_outputs(space):
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.sum(space.nvec))
    return int(space.n)


def _branch(q, n_joints):
    return q.view(-1, n_joints, 3)


def bike_groups(lanes):
    """bike obs = [speed] + nearest-hole distance in ego-relative lane slots (offsets -(w//2) .. w - w//2 - 1, w = lanes + lanes//2).
    Left head: speed, ego slot and the slots left of the bike. Right head: speed, ego slot and the slots to its right."""
    width = lanes + lanes // 2
    start = -(width // 2)
    idx = {off: 1 + i for i, off in enumerate(range(start, start + width))}
    return {"left": [0] + [i for off, i in idx.items() if off <= 0], "right": [0] + [i for off, i in idx.items() if off >= 0]}


OBS_GROUPS = {"CartPole-v1": {"cart": [0, 1], "pole": [2, 3]},
              "HalfCheetah-v4": {"pose": list(range(0, 8)), "velocity": list(range(8, 17))}}


class AuxHeads(nn.Module):
    """One linear Q head per observation group on the Q-network's last hidden layer (private-generator init)."""

    def __init__(self, feature_dim, n_out, groups, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed + 20_000)
        self.groups = [list(g) for g in groups.values()]
        self.heads = nn.ModuleList()
        for _ in self.groups:
            layer = nn.utils.skip_init(nn.Linear, feature_dim, n_out)
            bound = 1.0 / np.sqrt(feature_dim)
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=gen)
                layer.bias.uniform_(-bound, bound, generator=gen)
            self.heads.append(layer)

    def scores(self, q_network, obs, g):
        masked = torch.zeros_like(obs)
        masked[..., self.groups[g]] = obs[..., self.groups[g]]
        with torch.no_grad():  # heads do not train the trunk
            feat = q_network.network[:-1](masked)
        return self.heads[g](feat)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
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
    csv_file = None
    if args.csv_path:  # [added]
        csv_file = open(args.csv_path, "w", buffering=1)
        csv_file.write("global_step,episodic_return,clipped_return\n")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    BIKE_KW.update(lanes=args.lanes, solutions=args.solutions)
    eval_file = None
    if args.eval_every > 0 and args.eval_csv:
        eval_file = open(args.eval_csv, "w", buffering=1)
        eval_file.write("global_step,greedy_return\n")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    space = envs.single_action_space
    factored = isinstance(space, gym.spaces.MultiDiscrete)
    assert isinstance(space, gym.spaces.Discrete) or factored, "only discrete action space is supported"
    n_joints = int(len(space.nvec)) if factored else None

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    groups = (bike_groups(args.lanes) if args.env_id == "bike_v2.1" else OBS_GROUPS[args.env_id]) if args.heads == "groups" \
        else {f"obs{i}": [i] for i in range(obs_dim)}
    heads = AuxHeads(q_network.network[-1].in_features, _n_outputs(space), groups, args.seed)
    heads_optimizer = optim.Adam(heads.parameters(), lr=args.learning_rate)
    conf_file = None
    if args.conf_log_path:
        conf_file = open(args.conf_log_path, "w", buffering=1)
        conf_file.write("global_step,m,lr_eff\n")

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # [bike grid] shared greedy evaluation: no exploration, fixed fresh boards, own env (not a training step)
        if eval_file is not None and global_step > 0 and global_step % args.eval_every == 0:
            from common_bike import greedy_return

            def _greedy(o):
                with torch.no_grad():
                    o_t = torch.as_tensor(o, dtype=torch.float32, device=device)[None]
                    q_e = q_network(o_t)
                    if args.mix_lambda > 0:  # same mixed policy as in training
                        q_e = (1 - args.mix_lambda) * q_e + args.mix_lambda * torch.stack(
                            [heads.scores(q_network, o_t, g) for g in range(len(heads.heads))]).mean(0)
                    return int(torch.argmax(q_e, dim=1))

            eval_file.write(f"{global_step},{greedy_return(_greedy, args.lanes, args.solutions, args.eval_episodes, args.eval_seed0)}\n")

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.as_tensor(obs, dtype=torch.float32, device=device))
            if args.mix_lambda > 0:  # [ConfGate-mix] heads take part in the action choice (no gradient, no random draw)
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    q_heads = torch.stack([heads.scores(q_network, o_t, g) for g in range(len(heads.heads))]).mean(0)
                q_values = (1 - args.mix_lambda) * q_values + args.mix_lambda * q_heads
            if factored:
                actions = _branch(q_values, n_joints).argmax(-1).cpu().numpy()
            else:
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    if csv_file is not None:  # [added]
                        _r = float(np.squeeze(info["episode"]["r"]))
                        csv_file.write(f"{global_step},{_r},{_r}\n")
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    nxt = target_network(data.next_observations)
                    if factored:
                        target_max = _branch(nxt, n_joints).max(-1).values.mean(-1)
                    else:
                        target_max, _ = nxt.max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                q = q_network(data.observations)
                if factored:
                    old_val = _branch(q, n_joints).gather(-1, data.actions.long().unsqueeze(-1)).squeeze(-1).mean(-1)
                else:
                    old_val = q.gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # [ConfGate] m from the heads' greedy agreement on this batch; scale the lr for this update
                with torch.no_grad():
                    greedy = []
                    for g in range(len(heads.heads)):
                        qg = heads.scores(q_network, data.observations, g)
                        greedy.append((_branch(qg, n_joints) if factored else qg).argmax(-1))
                    votes = F.one_hot(torch.stack(greedy), 3 if factored else _n_outputs(space)).sum(0).float()
                    k_heads = len(greedy)  # mean pairwise agreement; for 2 heads exactly 1[a_1 = a_2]
                    m = float((votes * (votes - 1)).sum(-1).div(k_heads * (k_heads - 1)).mean())
                lr_eff = args.learning_rate * (args.lr_floor + args.lr_slope * m)
                for group in optimizer.param_groups:
                    group["lr"] = lr_eff
                if conf_file is not None and global_step % 100 == 0:
                    conf_file.write(f"{global_step},{m},{lr_eff}\n")

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # [ConfGate] fit the heads to the same TD target from their partial views (trunk detached)
                aux_loss = 0.0
                for g in range(len(heads.heads)):
                    qg = heads.scores(q_network, data.observations, g)
                    if factored:
                        qa = _branch(qg, n_joints).gather(-1, data.actions.long().unsqueeze(-1)).squeeze(-1).mean(-1)
                    else:
                        qa = qg.gather(1, data.actions).squeeze()
                    aux_loss = aux_loss + F.smooth_l1_loss(qa, td_target)
                heads_optimizer.zero_grad()
                aux_loss.backward()
                heads_optimizer.step()

            # update target network
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
