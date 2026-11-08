"""Adapted from searl/neuroevolution/evaluation_td3.py (MPEvaluation). Two
changes from the original, both forced by moving off the abandoned
gym==0.17.1 / mujoco-py stack onto modern gymnasium + the `mujoco` bindings
(the original repo's environment layer will not install on current systems --
mujoco-py needs the discontinued MuJoCo 2.0 binary):
  1. gymnasium's 5-tuple step() (obs, reward, terminated, truncated, info)
     replaces gym's 4-tuple; `done_bool` (which the original derived from
     "did we hit env._max_episode_steps") becomes simply `terminated`, since
     gymnasium's `truncated` flag *is* that time-limit signal.
  2. no multiprocessing.Pool -- population members are evaluated in a plain
     Python loop, which is fine at the population sizes used here.
The fitness/transition-collection logic itself (action rescale from the
actor's tanh [-1,1] output to the env's action bounds, exploration noise,
Transition bookkeeping) is unchanged.
"""
import numpy as np
import torch
import gymnasium as gym

from utils import to_tensor, Transition


def make_env(env_name):
    return gym.make(env_name)


def rescale_action(action, env):
    step_action = (action + 1) / 2  # [-1, 1] => [0, 1]
    step_action *= (env.action_space.high - env.action_space.low)
    step_action += env.action_space.low
    return step_action


def rollout_episode(env, actor_net, seed, exploration_noise=False, start_phase=False, greedy=False):
    """Runs one episode, returns (episode_return, transitions, n_frames)."""
    state, _ = env.reset(seed=int(seed))
    t_state = to_tensor(state).unsqueeze(0)
    terminated = truncated = False
    episode_return = 0.0
    transitions = []
    n_frames = 0

    with torch.no_grad():
        while not (terminated or truncated):
            if start_phase:
                action = to_tensor(env.action_space.sample())
            else:
                action = actor_net(t_state)
            action = action.clamp(-1, 1).data.numpy().flatten()

            if exploration_noise is not False and not greedy:
                action = action + exploration_noise * np.random.randn(*action.shape)
                action = np.clip(action, -1, 1)

            step_action = rescale_action(action, env)
            next_state, reward, terminated, truncated, _ = env.step(step_action)
            done_bool = float(terminated)

            transitions.append(Transition(state, action, next_state,
                                           np.array([reward]), np.array([done_bool]).astype("uint8")))
            t_state = to_tensor(next_state).unsqueeze(0)
            state = next_state
            episode_return += reward
            n_frames += 1

    return episode_return, transitions, n_frames


def evaluate_individual(individual, cfg, env, num_episodes, seed_base, exploration_noise=False, start_phase=False):
    individual.actor.eval()
    fitness_list = []
    all_transitions = []
    total_frames = 0
    episodes = 0
    while episodes < num_episodes or total_frames < cfg.eval.min_eval_steps:
        ret, transitions, n_frames = rollout_episode(env, individual.actor, seed_base + episodes,
                                                       exploration_noise=exploration_noise, start_phase=start_phase)
        fitness_list.append(ret)
        all_transitions.extend(transitions)
        total_frames += n_frames
        episodes += 1
    return fitness_list, all_transitions, total_frames


def evaluate_population(population, cfg, env, replay_memory, exploration_noise, total_frames_so_far, rng):
    start_phase = total_frames_so_far <= cfg.rl.start_timesteps
    mean_fitness = np.zeros(len(population))
    var_fitness = np.zeros(len(population))
    eval_frames = 0

    for idx, indi in enumerate(population):
        seed = int(rng.randint(0, 100_000))
        fitness_list, transitions, n_frames = evaluate_individual(
            indi, cfg, env, cfg.eval.eval_episodes, seed, exploration_noise=exploration_noise,
            start_phase=start_phase)

        mean_fitness[idx] = np.mean(fitness_list)
        var_fitness[idx] = np.var(fitness_list)
        eval_frames += n_frames
        replay_memory.add(transitions)

        indi.fitness.append(mean_fitness[idx])
        indi.improvement = indi.fitness[-1] - indi.fitness[-2] if len(indi.fitness) > 1 else indi.fitness[-1]

    return mean_fitness, var_fitness, eval_frames


def test_individual(individual, cfg, env, num_episodes, seed):
    fitness_list, _, _ = evaluate_individual(individual, cfg, env, num_episodes, seed, exploration_noise=False)
    return float(np.mean(fitness_list))
