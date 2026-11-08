# SEARL vs HPO methods vs fixed-hyperparameter baselines

`figures/multi_benchmark.png` (`make_multi_figure.py`): 2x2 — HalfCheetah-v4, bike_env v2.1, CartPole-v1 this
session, CartPole-v1 previous stored runs. x = total environment steps (log); mean +/- std. Solid = HPO /
adaptive method (SEARL, random search, HyperController, PB2, confidence gates); dashed = fixed hyperparameters
(CleanRL DQN/PPO). Every learned panel, including HalfCheetah, uses discrete actions: SEARL DQN, CleanRL DQN,
and CleanRL PPO. HalfCheetah-v4 keeps the MuJoCo body, but each of its 6 joints is restricted to -1, 0, or +1
(18 outputs), trained for 500k steps (`run_halfcheetah_discrete.sh`, results in `results_halfcheetah/`).
The previous CartPole panel is the stored conf-gate comparison and is not recomputed.

Pong and Enduro are left out of the figure. They are discrete, and the runs exist, but at these budgets they do
not produce a learning curve: every Pong method stays near −21 (SEARL 60k agent steps; CleanRL DQN / PPO 250k),
and Enduro's SEARL and PPO lines stay near zero. CleanRL DQN did learn Enduro (about 200–280) and is still omitted.

| | HalfCheetah | Pong / Enduro | bike v2.1 |
|---|---|---|---|
| SEARL | `../searl/searl_discrete.py --task gym` (MLP DQN), 500k, pop 4, 18 outputs | `../searl/searl_discrete.py --task atari` (CNN, original network/operators), 60k frames, pop 4 | `../searl/searl_discrete.py --task bike` (MLP), 480k, pop 4 |
| Random search | not on this figure | original DQN, 10 x 25k, x10 | MLP DQN, 10 x 48k, x10 |
| CleanRL DQN | `cleanrl_dqn_bike.py`, 500k, 18 outputs | `code/cleanrl_dqn_baseline` DQN, 250k | `cleanrl_dqn_bike.py`, 480k |
| CleanRL PPO | `ppo_hpo.py --hpo none`, 500k, 18 outputs | `ppo_hpo.py --hpo none`, 250k | same, 480k |
| HyperController, PB2 | not on this figure | `ppo_hpo.py --hpo HyperController/PB2` | same |
| Rainbow DQN (fixed HPs) | not run | original DQN code + dqn_config.yml, 100k, buffer 30k | n/a (CNN-only code) |

All SEARL lines on the figure come from `searl_discrete.py` (DQN). HalfCheetah results are in
`results_halfcheetah/`. `searl_mlp_dqn.py` in this folder is
superseded for SEARL (only its random-search part, `bike_rs`, is still used).

The HalfCheetah panel is only the three discrete runs above. `ppo_hpo.py` is CleanRL's
ppo_atari.py/ppo.py loop; the HyperController/PB2 schedulers are the verbatim files of `../hypercontroller`, called
once per PPO iteration like `rl_tests.py` does (train_batch_size bounds rescaled to the PPO batch).

Bike v2.1 board: 5 lanes, 240 rows, 8 solutions; one decision per row; all learners train on reward/20; returns
reported raw (perfect = 4900). Metrics: SEARL/random search on bike are greedy test returns (SEARL: exploration-free
test of its best individual each epoch, not counted in steps); CleanRL/PPO curves are training returns.
Caveat: this board is easy -- a greedy MLP-DQN policy scores 4900 after ~1.3k random transitions, so bike mostly
measures how early each method starts learning, not hyperparameter quality. Harder boards (e.g. 10 lanes, 1 solution)
exist in `bike_env/code/v2.1_code`.

Run: `run_ppo_suite.sh`, `run_cleanrl_dqn_bike.sh`, `run_bike_searl_rs.sh` (random search part only; this folder, env `cleanrl_dqn_repro`),
`../searl_dqn/reproduction/run_rainbow_baseline.sh` (env `searl_dqn_repro`), `../searl/run_discrete.sh` (SEARL), then `make_multi_figure.py`.
