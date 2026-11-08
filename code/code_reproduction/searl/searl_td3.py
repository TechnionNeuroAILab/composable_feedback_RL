"""Sequential, single-process port of searl/neuroevolution/searl_td3.py
(SEARLforTD3). The original spawns a `multiprocessing.Pool` of `nevo.worker`
processes plus a separate replay-memory-manager process reached through
`multiprocessing.Manager` queues, so that population evaluation/training can
run in parallel across a cluster. This reproduction evaluates/trains the
population in a plain Python loop instead -- at the population sizes used
here (<= 10) that costs wall-clock time but changes nothing about the
algorithm: population initialization, tournament selection, mutation,
shared-experience-replay training, and the epoch/frame bookkeeping are the
same steps in the same order as the original `evolve_population`.
"""
import copy
import time

import numpy as np
import torch

from individual_td3 import Individual
from replay_memory import ReplayMemory
from evaluation import make_env, evaluate_population, test_individual
from mutation import Mutations
from tournament_selection import TournamentSelection
from td3_training import train_population


class SEARLforTD3:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log

        torch.manual_seed(cfg.seed.torch)
        np.random.seed(cfg.seed.numpy)
        self.rng = np.random.RandomState(cfg.seed.evaluation)

        self.env = make_env(cfg.env.name)
        cfg.set_attr("action_dim", self.env.action_space.shape[0])
        cfg.set_attr("state_dim", self.env.observation_space.shape[0])

        self.replay_memory = ReplayMemory(capacity=cfg.train.replay_memory_size, batch_size=cfg.rl.batch_size)
        self.tournament = TournamentSelection(config=cfg)
        self.mutation = Mutations(config=cfg)

    def initial_population(self):
        population = []
        for idx in range(self.cfg.nevo.population_size):
            actor_config = copy.deepcopy(self.cfg.actor.get_dict)
            critic_config = copy.deepcopy(self.cfg.critic.get_dict)
            rl_config = copy.deepcopy(self.cfg.rl)
            indi = Individual(state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                               actor_config=actor_config, critic_config=critic_config,
                               rl_config=rl_config, index=idx, td3_double_q=self.cfg.train.td3_double_q)
            population.append(indi)
        return population

    def evolve_population(self, population, history):
        num_frames = 0
        epoch = 0
        # Opt-in. The default reproduction leaves this at 0 and does not
        # change the 50k-frame history schema or spend extra evaluation.
        heldout_interval = int(getattr(self.cfg.eval, "heldout_interval", 0) or 0)
        heldout_episodes = int(getattr(self.cfg.eval, "heldout_episodes", 10) or 10)
        next_mark = heldout_interval
        heldout_seed = int(self.cfg.eval.test_seed) + 100_000

        while num_frames < self.cfg.train.num_frames:
            epoch += 1
            t0 = time.time()

            mean_fitness, var_fitness, eval_frames = evaluate_population(
                population, self.cfg, self.env, self.replay_memory,
                exploration_noise=self.cfg.eval.exploration_noise, total_frames_so_far=num_frames, rng=self.rng)
            num_frames += eval_frames

            best_idx = int(np.argmax(mean_fitness))
            record = {"epoch": epoch, "num_frames": num_frames,
                      "mean_fitness": float(np.mean(mean_fitness)),
                      "best_fitness": float(mean_fitness[best_idx]),
                      "time": time.time() - t0}
            finished = num_frames >= self.cfg.train.num_frames
            if heldout_interval > 0 and (num_frames >= next_mark or finished):
                replay_before = len(self.replay_memory.storage)
                record["heldout_return"] = test_individual(
                    population[best_idx], self.cfg, self.env, heldout_episodes, heldout_seed)
                if len(self.replay_memory.storage) != replay_before:
                    raise RuntimeError("held-out evaluation wrote transitions into the replay")
                while next_mark <= num_frames:
                    next_mark += heldout_interval
            history.append(record)
            message = (f"[SEARL] epoch={epoch} frames={num_frames} "
                       f"mean_fitness={np.mean(mean_fitness):.1f} best={mean_fitness[best_idx]:.1f}")
            if "heldout_return" in record:
                message += f" heldout={record['heldout_return']:.1f}"
            self.log(message)

            if finished:
                break

            if self.cfg.nevo.selection:
                elite, population = self.tournament.select(population)

            if self.cfg.nevo.mutation:
                population = self.mutation.mutation(population)

            if self.cfg.nevo.training:
                population = train_population(population, eval_frames, self.cfg, self.replay_memory)

        return history


def run_searl_td3(cfg, log=print):
    searl = SEARLforTD3(cfg, log=log)
    population = searl.initial_population()
    history = []
    searl.evolve_population(population, history)
    return history
