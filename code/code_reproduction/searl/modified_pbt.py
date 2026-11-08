"""'Modified PBT' baseline from SEARL Section 4.1: "we modify PBT to be able
to adapt the neural architecture ... performing the same mutations to the
activation function and neural network size as in SEARL ... In other
aspects, we follow Jaderberg et al. (2017) ... We train and evaluate a
population of 20 agents for 100 generations. In every generation, the
agents are trained and evaluated for 10 000 steps each."

This reuses SEARL's own building blocks (EvolvableMLP individuals,
TournamentSelection, the architecture/activation/parameter/rl-hyperparameter
Mutations) so the only real difference from searl_td3.py is the one the
paper calls out: **no shared experience replay** (each individual keeps its
own private replay memory, populated only from its own rollouts) and a
**fixed per-generation step budget** instead of SEARL's adaptive
j=0.5-of-frames-generated training fraction.

Note: the original repo's `Individual.clone()` has what looks like a copy-paste
bug -- it deep-copies `self.replay_memory` but assigns the copy back onto
`self` instead of onto the returned `clone` (searl/neuroevolution/components/
individual_td3.py, last few lines), which would leave every clone aliasing
the same replay memory object. We fix that here (`clone.replay_memory = ...`)
since giving each PBT worker an independent memory is the entire point of
this baseline.
"""
import copy
import time

import numpy as np
import torch

from individual_td3 import Individual
from replay_memory import ReplayMemory
from evaluation import make_env, evaluate_individual
from mutation import Mutations
from tournament_selection import TournamentSelection
from td3_training import update_parameters


class ModifiedPBT:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        torch.manual_seed(cfg.seed.torch)
        np.random.seed(cfg.seed.numpy)
        self.rng = np.random.RandomState(cfg.seed.evaluation)

        self.env = make_env(cfg.env.name)
        cfg.set_attr("action_dim", self.env.action_space.shape[0])
        cfg.set_attr("state_dim", self.env.observation_space.shape[0])

        self.tournament = TournamentSelection(config=cfg)
        self.mutation = Mutations(config=cfg)

    def initial_population(self):
        population = []
        for idx in range(self.cfg.nevo.population_size):
            actor_config = copy.deepcopy(self.cfg.actor.get_dict)
            critic_config = copy.deepcopy(self.cfg.critic.get_dict)
            rl_config = copy.deepcopy(self.cfg.rl)
            replay_memory = ReplayMemory(capacity=self.cfg.train.replay_memory_size, batch_size=self.cfg.rl.batch_size)
            indi = Individual(state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                               actor_config=actor_config, critic_config=critic_config,
                               rl_config=rl_config, index=idx, td3_double_q=self.cfg.train.td3_double_q,
                               replay_memory=replay_memory)
            population.append(indi)
        return population

    def run(self, population, history):
        num_frames = 0
        steps_per_gen = self.cfg.nevo.steps_per_generation

        for generation in range(1, self.cfg.nevo.generations + 1):
            t0 = time.time()
            mean_fitness = np.zeros(len(population))
            for idx, indi in enumerate(population):
                seed = int(self.rng.randint(0, 100_000))
                fitness_list, transitions, n_frames = evaluate_individual(
                    indi, self.cfg, self.env, num_episodes=1, seed_base=seed,
                    exploration_noise=self.cfg.eval.exploration_noise)
                indi.replay_memory.add(transitions)
                mean_fitness[idx] = np.mean(fitness_list)
                indi.fitness.append(mean_fitness[idx])
                num_frames += n_frames

            best_idx = int(np.argmax(mean_fitness))
            history.append({"epoch": generation, "num_frames": num_frames,
                             "mean_fitness": float(np.mean(mean_fitness)),
                             "best_fitness": float(mean_fitness[best_idx]),
                             "time": time.time() - t0})
            self.log(f"[ModifiedPBT] gen={generation} frames={num_frames} "
                     f"mean_fitness={np.mean(mean_fitness):.1f} best={mean_fitness[best_idx]:.1f}")

            if generation == self.cfg.nevo.generations:
                break

            elite, new_population = self.tournament.select(population)
            # give every cloned individual its own independent replay memory
            # (fixes the upstream aliasing bug described in the module docstring)
            for indi in new_population:
                indi.replay_memory = copy.deepcopy(indi.replay_memory)
            population = self.mutation.mutation(new_population)

            for indi in population:
                if len(indi.replay_memory) < self.cfg.rl.batch_size:
                    continue
                update_parameters(indi, indi.replay_memory, steps_per_gen)

        return history


def run_modified_pbt(cfg, log=print):
    pbt = ModifiedPBT(cfg, log=log)
    population = pbt.initial_population()
    history = []
    pbt.run(population, history)
    return history
