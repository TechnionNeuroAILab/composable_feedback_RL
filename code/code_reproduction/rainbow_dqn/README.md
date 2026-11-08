# Rainbow DQN, ported from Kaixhin/Rainbow

Port of https://github.com/Kaixhin/Rainbow (commit 1745b18, MIT): dueling, noisy (factorised, std 0.1), categorical C51
(51 atoms), Double-Q, n-step (n=3) returns and proportional prioritized replay (segment tree, omega 0.5, beta 0.4 -> 1).

| file | origin |
|---|---|
| `model.py` | Kaixhin `model.py`; NoisyLinear verbatim. Adds an `mlp` trunk (vector observations) and `branches` (one dueling C51 head per action branch) |
| `agent.py` | Kaixhin `agent.py`; `learn()` is the original update, generalised over branches (branches=1 is identical) |
| `memory.py` | Kaixhin `memory.py`; transition layout is built per instance (uint8 frames or float32 vectors, action per branch); one `np.array` copy so torch can read unaligned float32 fields |
| `env_atari.py` | Kaixhin `env.py` on `ale_py` instead of the unmaintained `atari_py` (same preprocessing) |
| `env_tasks.py` | new: the vector tasks of multi_benchmark (bike v2.1, CartPole, Acrobot, LunarLander, HalfCheetah/Hopper with one 3-way torque choice per joint, Freeway RAM, FrozenLake) |
| `main.py` | Kaixhin `main.py` loop; adds task selection, reward scaling, CSV logging; no validation memory / separate eval pass |
| `test_port.py` | equivalence tests against a clone of the original |

## Checks (`python test_port.py --ref <path to Kaixhin/Rainbow clone>`)
1. one `learn()` step on identical weights, batch and random numbers: priorities and parameters differ by exactly 0
2. identical append sequence and RNG: the sampled batch (indices, states, actions, n-step returns, next states, terminals,
   importance weights) differs by exactly 0 (pixel mode, history 4, n 3)
3. 3 branches carrying identical data give the single-branch loss (max difference 6e-5: each branch draws its own noise)
The Atari path was smoke-tested on Pong (runs, episodes logged); it is slow on a busy shared GPU (tiny host->device copies
wait for the queue, ~15 steps/s): use `--disable-cuda`.

## Benchmark runs (`run_discrete_tasks.sh` -> `results/<task>/seed<k>.csv`, one row per finished episode)
Kaixhin's Atari defaults (lr 6.25e-5, replay every 4 steps, target update 8000, learn start 20k, 1M replay) did not learn
CartPole in a 60k-step test, so the budget-scaled settings below are used for every task: replay every 2 steps, lr 2.5e-4,
target update every 1000 steps, learn start 2000, replay 100k, batch 32, MLP trunk 128 then 512-unit noisy streams.
The C51 support must cover the discounted return, so per task the reward is multiplied by a scale and the support is set
(reported returns are always raw):

| task | scale | support | steps x seeds |
|---|---|---|---|
| CartPole-v1 | 0.1 | [0, 10] | 1M x 10 |
| bike v2.1 | 0.005 (raw reward) | [-10, 10] | 480k x 3 |
| Acrobot-v1 | 0.1 | [-10, 0] | 480k x 3 |
| LunarLander-v2 | 0.05 | [-10, 10] | 480k x 3 |
| HalfCheetah-v4 (18 outputs) | 0.05 | [-5, 10] | 500k x 3 |
| Hopper-v4 (9 outputs) | 0.03 | [-1, 10] | 480k x 3 |
| ALE/Freeway-ram-v5 | 1 (obs / 255) | [0, 10] | 480k x 3 |
| FrozenLake-v1 | 1 | [0, 1] | 100k x 3 |

Logged returns are training-episode returns (the noisy-net policy, like the CleanRL DQN curves). Truncation at the time limit
is treated as terminal, as in the CleanRL baselines.
