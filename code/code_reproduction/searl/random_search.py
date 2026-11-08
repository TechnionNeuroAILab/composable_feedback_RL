"""Random Search baseline, SEARL Section 4.1: "We sample 20 random
configurations and train TD3 using each configuration without any online
architecture or hyperparameter changes for 1M frames. We select the
configuration that resulted in the best training performance."

Search space is Table 1's "Random Search" column: learning rate log-uniform
in [1e-5, 5e-3], activation in {relu, tanh, elu}, 1-3 hidden layers of 64-500
units each (same width sampled for every layer of a given config, matching
how searl_td3_config.yml itself specifies a single `hidden_size` list).
"""
import copy

import numpy as np

from td3_baseline import TD3Agent


def sample_config(rng, base_cfg, num_configs_idx):
    cfg = copy.deepcopy(base_cfg)

    lr = float(np.exp(rng.uniform(np.log(1e-5), np.log(5e-3))))
    activation = rng.choice(["relu", "tanh", "elu"])
    n_layers = int(rng.choice([1, 2, 3]))
    width = int(rng.uniform(64, 500))
    hidden_size = [width] * n_layers

    cfg.rl.lr_actor = lr
    cfg.rl.lr_critic = lr
    cfg.actor.hidden_size = list(hidden_size)
    cfg.actor.activation = activation
    cfg.critic.hidden_size = list(hidden_size)
    cfg.critic.activation = activation
    cfg.expt.experiment_name = f"rs_config_{num_configs_idx}"
    return cfg


def run_random_search(base_cfg, num_configs, frames_per_config, seed, log=print):
    rng = np.random.RandomState(seed)
    results = []
    for i in range(num_configs):
        cfg = sample_config(rng, base_cfg, i)
        cfg.td3.max_timesteps = frames_per_config
        cfg.seed.torch = seed * 1000 + i
        cfg.seed.numpy = seed * 1000 + i
        agent = TD3Agent(cfg)
        history = agent.perform_learning(log=log)
        final_reward = history[-1]["reward"]
        results.append({"config_idx": i, "lr": cfg.rl.lr_actor, "activation": cfg.actor.activation,
                         "hidden_size": cfg.actor.hidden_size, "final_reward": final_reward, "history": history})
        log(f"[RandomSearch] config {i}: lr={cfg.rl.lr_actor:.2e} act={cfg.actor.activation} "
            f"hidden={cfg.actor.hidden_size} final_reward={final_reward:.1f}")

    best = max(results, key=lambda r: r["final_reward"])
    log(f"[RandomSearch] best config: {best['config_idx']} (final_reward={best['final_reward']:.1f})")
    return results, best
