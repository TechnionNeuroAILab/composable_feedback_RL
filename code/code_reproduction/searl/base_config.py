"""Builds the config tree used by searl_td3.py / modified_pbt.py /
random_search.py, mirroring the structure (and, where scale allows, the
values) of the original repo's configs/searl_td3_config.yml and
configs/td3_config.yml. Section 4.2 of the paper and those two YAML files are
the source for every value below; the only intentional departures are the
*_REDUCED_SCALE_* numbers, called out where they occur, which cut run
length/population size so the whole reproduction finishes in a practical
amount of time on this machine instead of the paper's 2M-20M env-step
budgets (see run_reproduction.py's module docstring for the exact factors).
"""
from config import make_config

ENV_NAME = "HalfCheetah-v4"  # paper uses HalfCheetah-v2 (mujoco-py); v4 is the
                              # modern mujoco-bindings equivalent, see README.


def base_dict(seed=0):
    return {
        "expt": {"project_name": "searl_repro", "session_name": "halfcheetah", "experiment_name": "default"},
        "nevo": {
            "population_size": 3,    # paper: 20
            "tournament_size": 3,
            "selection": True,
            "mutation": True,
            "training": True,
            "elitism": True,
            "generations": 67,             # modified-PBT only; paper: 100
            "steps_per_generation": 200,   # modified-PBT only; paper: 10_000
        },
        "mutation": {
            "no_mutation": 0.2,
            "parameters": 0.2,
            "architecture": 0.2,
            "activation": 0.2,
            "rl_hyperparam": 0.2,
            "rl_hp_selection": ["lr_actor", "lr_critic"],
            "new_layer_prob": 0.2,
            "mutation_sd": 0.1,
        },
        "train": {
            "replay_memory_size": 1_000_000,
            "num_frames": 50_000,    # paper: 2_000_000 (see run_reproduction.py)
            "td3_double_q": True,
            "min_train_steps": 250,
        },
        "rl": {
            "train_frames_fraction": 0.5,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 100,
            "lr_actor": 0.001,
            "lr_critic": 0.001,
            "clip_grad_norm": 100,
            "td3_policy_noise": 0.2,
            "td3_noise_clip": 0.5,
            "td3_update_freq": 2,
            "optimizer": "adam",
            "start_timesteps": 1,
        },
        "seed": {"replay_memory": seed, "evaluation": seed, "mutation": seed, "training": seed,
                 "torch": seed, "numpy": seed},
        "env": {"name": ENV_NAME},
        "eval": {"eval_episodes": 1, "min_eval_steps": 250, "exploration_noise": 0.1,
                 "test_episodes": 10, "test_seed": seed,
                 # 0 keeps the default reproduction from spending a held-out eval.
                 # run_reproduction.py --heldout-interval turns it on for matched runs.
                 "heldout_interval": 0, "heldout_episodes": 10},
        "actor": {"hidden_size": [128], "activation": "relu", "output_activation": "tanh",
                  "layer_norm": True, "output_vanish": False},
        "critic": {"hidden_size": [128], "activation": "relu", "output_activation": "linear",
                   "layer_norm": True, "output_vanish": True},
        # td3 section used only by TD3Agent (random-search / modified-PBT baseline trainer);
        # mirrors rl -> td3 field-name mapping from the original configs/td3_config.yml
        "td3": {
            "gamma": 0.99, "tau": 0.005, "lr_actor": 0.001, "lr_critic": 0.001, "batch_size": 100,
            "double_q": True, "clip_grad_norm": 100, "td3_policy_noise": 0.2, "td3_noise_clip": 0.5,
            "td3_update_freq": 2, "optimizer": "adam", "rm_capacity": 1_000_000, "eval_freq": 5_000,
            "start_timesteps": 10_000, "exploration_noise": 0.1, "eval_episodes": 10,
            "max_timesteps": 200_000,  # paper RS: 1_000_000 per config (see run_reproduction.py)
        },
    }


def make_base_config(seed=0):
    return make_config(base_dict(seed))
