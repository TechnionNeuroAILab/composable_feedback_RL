"""Standalone single-agent TD3, adapted from searl/rl_algorithms/td3.py
(the `TD3` class / `start_TD3_training`). Used as the per-configuration
trainer for the "random search" baseline in Figure 2a: each sampled
(architecture, activation, learning-rate) configuration trains one of these
agents, with a fixed network and no online hyperparameter/architecture
adaptation, exactly as Section 4.1 ("Random Search") describes.

Unlike searl/neuroevolution/training_td3.py (used inside SEARL's
evolutionary loop, which always recreates fresh target networks/optimizers
because the architecture may have just been mutated), this baseline keeps
persistent target networks and optimizers across the whole run -- matching
the original rl_algorithms/td3.py, where `reset_target`/`recreate_optim`
default to False.

API changes from the original are the same ones as evaluation.py: gymnasium's
5-tuple step() replacing gym's 4-tuple, and `terminated` replacing the
`done_bool = 0 if hit time-limit else done` special-casing.
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from evolvable_mlp import EvolvableMLP
from evaluation import make_env, rescale_action
from utils import to_tensor, Transition
from replay_memory import ReplayMemory
from td3_training import get_optimizer


class TD3Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = make_env(cfg.env.name)
        cfg.set_attr("action_dim", self.env.action_space.shape[0])
        cfg.set_attr("state_dim", self.env.observation_space.shape[0])

        self.actor = EvolvableMLP(num_inputs=cfg.state_dim, num_outputs=cfg.action_dim, **cfg.actor.get_dict)
        self.actor_target = type(self.actor)(**self.actor.init_dict)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = EvolvableMLP(num_inputs=cfg.state_dim + cfg.action_dim, num_outputs=1, **cfg.critic.get_dict)
        self.critic_1_target = type(self.critic_1)(**self.critic_1.init_dict)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = EvolvableMLP(num_inputs=cfg.state_dim + cfg.action_dim, num_outputs=1, **cfg.critic.get_dict)
        self.critic_2_target = type(self.critic_2)(**self.critic_2.init_dict)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        Opti = get_optimizer(cfg.td3.optimizer)
        self.actor_optim = Opti(self.actor.parameters(), lr=cfg.td3.lr_actor)
        self.critic_1_optim = Opti(self.critic_1.parameters(), lr=cfg.td3.lr_critic)
        self.critic_2_optim = Opti(self.critic_2.parameters(), lr=cfg.td3.lr_critic)

        self.replay_memory = ReplayMemory(capacity=int(cfg.td3.rm_capacity), batch_size=cfg.td3.batch_size)

    def select_action(self, t_state):
        with torch.no_grad():
            return self.actor(t_state).clamp(-1, 1).data.numpy().flatten()

    def train(self, iterations):
        args = self.cfg.td3
        for it in range(iterations):
            transitions = self.replay_memory.sample(args.batch_size)

            state = torch.stack([torch.Tensor(t.state) for t in transitions], dim=0)
            action = torch.stack([torch.Tensor(t.action) for t in transitions], dim=0)
            next_state = torch.stack([torch.Tensor(t.next_state) for t in transitions], dim=0)
            reward = torch.stack([torch.Tensor(t.reward) for t in transitions], dim=0)
            done = 1 - torch.stack([torch.Tensor(t.done) for t in transitions], dim=0)

            with torch.no_grad():
                noise = (torch.randn_like(action) * args.td3_policy_noise).clamp(-args.td3_noise_clip,
                                                                                  args.td3_noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
                target_Q1 = self.critic_1_target(torch.cat([next_state, next_action], 1))
                target_Q2 = self.critic_2_target(torch.cat([next_state, next_action], 1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * args.gamma * target_Q)

            current_Q1 = self.critic_1(torch.cat([state, action], 1))
            current_Q2 = self.critic_2(torch.cat([state, action], 1))

            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optim.zero_grad()
            critic_loss_1.backward()
            for p in self.critic_1.parameters():
                p.grad.data.clamp_(max=args.clip_grad_norm)
            self.critic_1_optim.step()

            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optim.zero_grad()
            critic_loss_2.backward()
            for p in self.critic_2.parameters():
                p.grad.data.clamp_(max=args.clip_grad_norm)
            self.critic_2_optim.step()

            if it % args.td3_update_freq == 0:
                actor_loss = -self.critic_1(torch.cat([state, self.actor(state)], 1))
                actor_loss = torch.mean(actor_loss)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                for p in self.actor.parameters():
                    p.grad.data.clamp_(max=args.clip_grad_norm)
                self.actor_optim.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    def evaluate_policy(self, eval_episodes):
        episode_rewards = []
        for _ in range(eval_episodes):
            state, _ = self.env.reset()
            t_state = to_tensor(state).unsqueeze(0)
            terminated = truncated = False
            episode_reward = 0.0
            with torch.no_grad():
                while not (terminated or truncated):
                    action = self.select_action(t_state)
                    step_action = rescale_action(action, self.env)
                    next_state, reward, terminated, truncated, _ = self.env.step(step_action)
                    t_state = to_tensor(next_state).unsqueeze(0)
                    episode_reward += reward
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards))

    def perform_learning(self, log=print, history=None):
        history = [] if history is None else history
        cfg = self.cfg.td3
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        terminated = truncated = True
        episode_reward = episode_timesteps = 0
        state = None
        t_state = None

        while total_timesteps < cfg.max_timesteps:
            if terminated or truncated:
                if total_timesteps != 0:
                    self.train(episode_timesteps)
                if timesteps_since_eval >= cfg.eval_freq or total_timesteps == 0:
                    timesteps_since_eval %= cfg.eval_freq
                    test_reward = self.evaluate_policy(cfg.eval_episodes)
                    history.append({"num_frames": total_timesteps, "reward": test_reward})
                    log(f"[TD3 cfg={self.cfg.expt.experiment_name}] frames={total_timesteps} reward={test_reward:.1f}")

                state, _ = self.env.reset()
                t_state = to_tensor(state).unsqueeze(0)
                terminated = truncated = False
                episode_reward = 0.0
                episode_timesteps = 0
                episode_num += 1

            if total_timesteps < cfg.start_timesteps:
                action = to_tensor(self.env.action_space.sample()).clamp(-1, 1).data.numpy().flatten()
            else:
                action = self.select_action(t_state)
                if cfg.exploration_noise is not False:
                    action = action + cfg.exploration_noise * np.random.randn(*action.shape)
                    action = np.clip(action, -1, 1)

            step_action = rescale_action(action, self.env)
            next_state, reward, terminated, truncated, _ = self.env.step(step_action)
            done_bool = float(terminated)

            self.replay_memory.add(Transition(state, action, next_state,
                                               np.array([reward]), np.array([done_bool]).astype("uint8")))
            t_state = to_tensor(next_state).unsqueeze(0)
            state = next_state

            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        test_reward = self.evaluate_policy(cfg.eval_episodes)
        history.append({"num_frames": total_timesteps, "reward": test_reward})
        return history
