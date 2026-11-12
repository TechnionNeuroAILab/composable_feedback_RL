# [Feature-specific channels, single agent] CleanRL DQN (same loop and defaults as cleanrl_dqn_confgate.py / cleanrl_dqn_bike.py)
# with the Q-network split into additive channels, one per observation group plus one channel on the full observation.
# See paper/figures/proposal/all_options.png. Variants (--variant):
#
#   additive  Q = sum_i Q_i, trained on the ordinary TD error delta; no control loop (architecture control for the others).
#   ac        (A+C) same network and same shared delta; the per-channel errors
#                 delta_i = r/N + gamma * Q_i^tgt(s', a*) - Q_i(s, a),   sum_i delta_i = delta,
#             only drive the loop: each channel's step size alpha_i follows the signal-to-noise of its delta_i.
#   flite     (F-lite) as ac, but channel i is trained on its own blended error  e_i = (1 - beta) * delta + beta * delta_i.
#   f         (F) successor-feature channels: channel g predicts the discounted future of its own (normalised) observation
#             group, psi_g(s, a); Q = sum_g w_g . psi_g + Q_res. Several errors train:
#                 delta^psi_g = phi_g(s') + gamma * psi_g^tgt(s', a*) - psi_g(s, a)      -> encoder of channel g
#                 delta^r     = r - sum_g w_g . phi_g(s')                                  -> the reward weights w_g (shared)
#                 delta_res   = r + gamma * Q^tgt(s', a*) - Q(s, a)                        -> the residual channel only
#             and the loop sets each channel's step size from the SNR of its own error.
#
# Loop rule (all variants with a loop): per channel, EMA of the batch mean (mu_i) and batch sd (sd_i) of its error;
# SNR_i = |mu_i| / sd_i; alpha_i = lr * clip((SNR_i / mean_j SNR_j) ** kappa, alpha_min, alpha_max).
# Factored actions (HalfCheetah: 6 joints x 3 torques): Q(s, a) = mean_j Q[j, a_j], as in the other CleanRL scripts.
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

from cleanrl_dqn_confgate import OBS_GROUPS, _branch, _n_outputs, linear_schedule, make_env
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    seed: int = 0
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10
    csv_path: str = ""
    """write global_step, episodic_return, clipped_return (same format as the other CleanRL scripts)"""
    chan_log_path: str = ""
    """write per-channel loop statistics every 1000 steps"""
    variant: str = "ac"
    """additive | ac | flite | f"""
    hidden: tuple = (64, 48)
    """hidden sizes of every channel (3 channels x (64, 48) ~ the parameter count of CleanRL's single (120, 84) MLP)"""
    beta: float = 0.5
    """[flite] weight of the channel's own error in its training error"""
    kappa: float = 1.0
    alpha_min: float = 0.25
    alpha_max: float = 2.0
    ema: float = 0.01
    """EMA rate of the loop's error statistics"""
    w_lr: float = 1e-3
    no_loop: bool = False
    """control: keep each channel's error composition but fix alpha_i = 1 (the SNR is still computed and logged)"""
    """[f] learning rate of the reward weights w_g"""


def mlp(n_in, hidden, n_out):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, n_out))


class Channels(nn.Module):
    """One sub-network per channel; channel i sees only its observation entries (others zeroed). Channel outputs:
    additive/ac/flite -> Q_i (n_out);  f -> psi_g (n_out x d_g) for the groups and Q_res (n_out) for the full channel."""

    def __init__(self, obs_dim, n_out, groups, hidden, sf):
        super().__init__()
        self.names = list(groups) + ["full"]
        self.idx = [list(g) for g in groups.values()] + [list(range(obs_dim))]
        self.sf, self.n_out = sf, n_out
        self.dims = [len(g) for g in groups.values()]
        outs = [n_out * d for d in self.dims] + [n_out] if sf else [n_out] * len(self.names)
        self.nets = nn.ModuleList([mlp(obs_dim, hidden, o) for o in outs])
        self.w = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in self.dims]) if sf else None

    def channel_outputs(self, x):
        outs = []
        for net, idx in zip(self.nets, self.idx):
            masked = torch.zeros_like(x)
            masked[:, idx] = x[:, idx]
            outs.append(net(masked))
        return outs

    def q_parts(self, x):
        """Per-channel Q contributions, each (B, n_out). For f also returns the psi tensors (B, n_out, d_g)."""
        outs = self.channel_outputs(x)
        if not self.sf:
            return outs, None
        psis = [o.view(-1, self.n_out, d) for o, d in zip(outs[:-1], self.dims)]
        parts = [p @ w for p, w in zip(psis, self.w)] + [outs[-1]]
        return parts, psis

    def forward(self, x):
        return torch.stack(self.q_parts(x)[0]).sum(0)


def take(q, actions, n_joints):
    """Q(s, a) for (B, n_out) Q-values: per-joint gather, mean over joints (factored) or plain gather."""
    if n_joints:
        return _branch(q, n_joints).gather(-1, actions.long().unsqueeze(-1)).squeeze(-1).mean(-1)
    return q.gather(1, actions).squeeze(1)


def take_vec(psi, actions, n_joints):
    """psi(s, a) for (B, n_out, d) successor features (factored: mean over joints of psi[j, a_j])."""
    if n_joints:
        d = psi.shape[-1]
        p = psi.view(psi.shape[0], n_joints, 3, d)
        idx = actions.long().view(-1, n_joints, 1, 1).expand(-1, -1, 1, d)
        return p.gather(2, idx).squeeze(2).mean(1)
    return psi.gather(1, actions.view(-1, 1, 1).expand(-1, 1, psi.shape[-1])).squeeze(1)


def greedy(q, n_joints):
    return _branch(q, n_joints).argmax(-1) if n_joints else q.argmax(1, keepdim=True)


class SNRLoop:
    """alpha_i = lr * clip((SNR_i / mean SNR) ** kappa, alpha_min, alpha_max), from EMAs of each channel's error."""

    def __init__(self, n, a):
        self.mu, self.sd, self.a = np.zeros(n), np.ones(n), a
        self.started = False

    def update(self, errors):
        mu = np.array([float(e.mean()) for e in errors])
        sd = np.array([float(e.std()) + 1e-8 for e in errors])
        if not self.started:
            self.mu, self.sd, self.started = mu, sd, True
        else:
            self.mu += self.a.ema * (mu - self.mu)
            self.sd += self.a.ema * (sd - self.sd)
        snr = np.abs(self.mu) / self.sd
        rel = snr / (snr.mean() + 1e-12)
        return np.clip(rel ** self.a.kappa, self.a.alpha_min, self.a.alpha_max), snr


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.variant in ("additive", "ac", "flite", "f")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, "")])
    space = envs.single_action_space
    n_joints = int(len(space.nvec)) if isinstance(space, gym.spaces.MultiDiscrete) else None
    n_out = _n_outputs(space)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    groups = OBS_GROUPS[args.env_id]
    sf = args.variant == "f"

    q_network = Channels(obs_dim, n_out, groups, args.hidden, sf).to(device)
    target_network = Channels(obs_dim, n_out, groups, args.hidden, sf).to(device)
    target_network.load_state_dict(q_network.state_dict())
    n_ch = len(q_network.nets)
    optimizer = optim.Adam([{"params": net.parameters()} for net in q_network.nets], lr=args.learning_rate)
    w_optimizer = optim.Adam(q_network.w.parameters(), lr=args.w_lr) if sf else None
    loop = SNRLoop(n_ch, args) if args.variant != "additive" else None
    phi_mu = phi_sd = None   # [f] observation normalisation, frozen from the random-action warm-up data

    csv_file = open(args.csv_path, "w", buffering=1) if args.csv_path else None
    if csv_file:
        csv_file.write("global_step,episodic_return,clipped_return\n")
    chan_file = open(args.chan_log_path, "w", buffering=1) if args.chan_log_path else None
    if chan_file:
        chan_file.write("global_step,channel,alpha,snr,err_mean,err_abs,cancellation,w_norm\n")

    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, space, device, handle_timeout_termination=False)
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([space.sample()])
        else:
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = greedy(q_values, n_joints).cpu().numpy()
            actions = actions if n_joints else actions[:, 0]

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    r = float(np.squeeze(info["episode"]["r"]))
                    print(f"global_step={global_step}, episodic_return={r:.1f}, SPS={int(global_step / (time.time() - start_time))}")
                    if csv_file:
                        csv_file.write(f"{global_step},{r},{r}\n")
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            if sf and phi_mu is None:
                o = torch.as_tensor(rb.next_observations[: (rb.buffer_size if rb.full else rb.pos)].reshape(-1, obs_dim))
                phi_mu, phi_sd = o.mean(0), o.std(0) + 1e-6
            data = rb.sample(args.batch_size)
            s, s2, a = data.observations, data.next_observations, data.actions
            r, d = data.rewards.flatten(), data.dones.flatten()
            with torch.no_grad():
                tparts, tpsis = target_network.q_parts(s2)
                a_star = greedy(torch.stack(tparts).sum(0), n_joints)            # greedy next action of the full target Q
                next_parts = [take(p, a_star, n_joints) for p in tparts]          # Q_i^tgt(s', a*)
                td_target = r + args.gamma * torch.stack(next_parts).sum(0) * (1 - d)
            parts, psis = q_network.q_parts(s)
            cur = [take(p, a, n_joints) for p in parts]                          # Q_i(s, a)
            q_sa = torch.stack(cur).sum(0)
            delta = (td_target - q_sa).detach()

            if not sf:
                n = len(cur)
                errs = [(r / n + args.gamma * (1 - d) * nx - c).detach() for nx, c in zip(next_parts, cur)]
                if args.variant == "flite":
                    train_err = [(1 - args.beta) * delta + args.beta * e for e in errs]
                    # surrogate whose gradient on channel i is -2 e_i dQ_i  (beta = 0 gives exactly the MSE gradient)
                    loss = -sum(2 * (e * c).mean() for e, c in zip(train_err, cur))
                else:
                    loss = F.mse_loss(q_sa, td_target)
            else:
                phi2 = (s2 - phi_mu) / phi_sd
                phis = [phi2[:, idx] for idx in q_network.idx[:-1]]
                errs, loss = [], 0.0
                for g, (psi, tpsi, ph) in enumerate(zip(psis, tpsis, phis)):
                    psi_sa = take_vec(psi, a, n_joints)
                    with torch.no_grad():
                        tgt = ph + args.gamma * (1 - d)[:, None] * take_vec(tpsi, a_star, n_joints)
                    err = tgt - psi_sa
                    errs.append(err.detach().mean(1))                            # one scalar per sample for the loop
                    loss = loss + (err ** 2).mean()
                loss = loss + F.mse_loss(cur[-1], (td_target - torch.stack(cur[:-1]).sum(0)).detach())   # residual only
                errs.append(delta)                                                # residual channel: delta_res
                r_err = r - sum(ph @ w for ph, w in zip(phis, q_network.w))      # reward error -> w_g (shared)
                w_optimizer.zero_grad()
                (r_err ** 2).mean().backward()
                w_optimizer.step()

            alphas = np.ones(n_ch)
            if loop is not None:
                a_loop, snr = loop.update(errs)
                if not args.no_loop:
                    alphas = a_loop
            for g, al in zip(optimizer.param_groups, alphas):
                g["lr"] = args.learning_rate * al
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if chan_file and global_step % 1000 == 0:
                tot = torch.stack(errs).sum(0) if not sf else None
                canc = float(tot.abs().mean() / (torch.stack(errs).abs().sum(0).mean() + 1e-8)) if tot is not None else float("nan")
                for i, name in enumerate(q_network.names):
                    wn = float(q_network.w[i].norm()) if sf and i < len(q_network.w) else float("nan")
                    chan_file.write(f"{global_step},{name},{alphas[i]:.4f},{(snr[i] if loop else 0):.4f},"
                                    f"{float(errs[i].mean()):.5f},{float(errs[i].abs().mean()):.5f},{canc:.4f},{wn:.4f}\n")

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                tp.data.copy_(args.tau * qp.data + (1.0 - args.tau) * tp.data)
    envs.close()
