# Port of Kaixhin/Rainbow main.py: the same Rainbow training loop (noise reset every replay_frequency steps, learning
# from learn_start, importance-weight annealing, target update every target_update steps, n-step prioritized
# replay). PORT changes (marked): ale_py environment, vector-task support, CSV logging of finished episodes, no
# validation-memory / separate evaluation pass (the figure uses training-episode returns like the CleanRL baselines).
import argparse
import os
import random
import time

import numpy as np
import torch

from agent import Agent
from memory import ReplayMemory

parser = argparse.ArgumentParser(description='Rainbow (ported)')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--task', type=str, default='atari', choices=['atari', 'gym'], help='PORT: pixels (Atari) or vector task')
parser.add_argument('--game', type=str, default='pong', help='ATARI game (ale_py ROM name, e.g. pong, road_runner)')
parser.add_argument('--env', type=str, default='CartPole-v1', help='PORT: vector task id (or "bike")')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames in Atari)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient', 'mlp'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--trunk-size', type=int, default=128, metavar='SIZE', help='PORT: width of the MLP trunk (mlp architecture)')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--reward-scale', type=float, default=1.0, help='PORT: vector tasks: rewards are multiplied by this before learning '
                    '(the C51 support [V-min, V-max] must cover the discounted return)')
parser.add_argument('--obs-scale', type=float, default=1.0, help='PORT: vector tasks: observations are multiplied by this')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--lanes', type=int, default=5, help='bike grid: lanes of bike_v2.1')
parser.add_argument('--solutions', type=int, default=8, help='bike grid: solutions of bike_v2.1')
parser.add_argument('--eval-every', type=int, default=0, help='bike grid: if > 0, greedy evaluation every EVAL_EVERY steps')
parser.add_argument('--eval-episodes', type=int, default=10)
parser.add_argument('--eval-seed0', type=int, default=5000)
parser.add_argument('--eval-csv', type=str, default='')
parser.add_argument('--csv-path', type=str, default='', help='PORT: write global_step,episodic_return,clipped_return per finished episode')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
use_cuda = torch.cuda.is_available() and not args.disable_cuda and args.task == 'atari'
args.device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
  torch.cuda.manual_seed(np.random.randint(1, 10000))

# Environment
if args.task == 'atari':
  from env_atari import Env
  env = Env(args)
  action_space, branches, obs_dim = env.action_space(), 1, None
  mem_kwargs = dict(state_shape=(84, 84), state_dtype=np.uint8, branches=1)
else:
  from env_tasks import GymTask
  env = GymTask(args.env, args.seed, args.reward_scale, args.obs_scale, args.lanes, args.solutions)
  args.history_length, args.architecture, args.reward_clip = 1, 'mlp', 0
  action_space, branches, obs_dim = env.n_actions, env.branches, env.obs_dim
  mem_kwargs = dict(state_shape=(obs_dim,), state_dtype=np.float32, branches=branches)
print({k: v for k, v in vars(args).items()}, flush=True)

dqn = Agent(args, action_space, branches, obs_dim)
mem = ReplayMemory(args, args.memory_capacity, **mem_kwargs)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

csv_file = None
if args.csv_path:
  csv_file = open(args.csv_path, 'w', buffering=1)
  csv_file.write('global_step,episodic_return,clipped_return\n')

eval_file = None
if args.eval_every > 0 and args.eval_csv:
  eval_file = open(args.eval_csv, 'w', buffering=1)
  eval_file.write('global_step,greedy_return\n')

# Training loop
dqn.train()
done, new_game, raw_return, learn_return = True, True, 0.0, 0.0
start = time.time()
for T in range(1, args.T_max + 1):
  if done:
    state = env.reset()
    if new_game:  # Atari: a life loss ends the training episode but not the game being scored
      raw_return, learn_return = 0.0, 0.0

  if T % args.replay_frequency == 0:
    dqn.reset_noise()  # Draw a new set of noisy weights

  action = dqn.act(state)  # Choose an action greedily (with noisy weights)
  next_state, reward, done = env.step(action)  # Step
  raw_return += reward
  if args.task == 'atari':
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
  else:
    reward = reward * args.reward_scale  # PORT
  learn_return += reward
  mem.append(state, action, reward, done)  # Append transition to memory

  new_game = env.true_done if args.task == 'atari' else done
  if csv_file is not None and new_game:
    csv_file.write('%d,%s,%s\n' % (T, raw_return, learn_return if args.task == 'atari' else raw_return))

  # Train
  if T >= args.learn_start:
    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight beta to 1

    if T % args.replay_frequency == 0:
      dqn.learn(mem)  # Train with n-step distributional double-Q learning

    # Update target network
    if T % args.target_update == 0:
      dqn.update_target_net()

  if eval_file is not None and T % args.eval_every == 0:  # bike grid: shared greedy evaluation (mean weights, no noise)
    from common_bike import greedy_return
    dqn.eval()
    eval_file.write('%d,%s\n' % (T, greedy_return(lambda o: dqn.act(torch.tensor(np.asarray(o, np.float32).reshape(-1) * args.obs_scale)),
                                                   args.lanes, args.solutions, args.eval_episodes, args.eval_seed0)))
    dqn.train()

  if T % 20000 == 0:
    print('T = %d / %d | %d steps/s' % (T, args.T_max, T / (time.time() - start)), flush=True)

  state = next_state

env.close()
if csv_file is not None:
  csv_file.close()
