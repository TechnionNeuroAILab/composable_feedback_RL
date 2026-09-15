# Composable PPO with locally-controlled channels

One PPO agent with:

- a shared encoder and shared actor;
- `K_max` value critics (one per channel);
- vector TD errors `delta_t in R^K`;
- one GAE advantage per channel (its own `gamma_k`, `lambda_k`);
- sparse routing `B in R^{M x K}` from `M` shared encoder/actor modules into
  the channels;
- soft activation gates `g_k in [0, 1]` representing recruited loops.

## Files

- `config.py` -- `ExperimentConfig` (architecture/PPO schedule/controller step
  sizes) and `smoke_config()`.
- `core.py` -- `ModularEncoder` (M parallel branches), `ChannelRouter` (`B`
  and `g`), `SharedActor`, `ChannelCritic` x K, `ComposablePPOAgent`, plus
  per-channel GAE/TD-error/explained-variance helpers.
- `local_controller.py` -- the decentralized controller (see below).
- `train.py` -- rollout collection, PPO update, and the CLI (`--smoke`).

## Architecture

The encoder is `M` independent small MLP branches producing feature blocks
`z_1, ..., z_M`. `B`'s column `k` (softplus + normalized to sum to 1) says
how channel `k` reads the `M` modules: `c_k = sum_m B_{mk} z_m` feeds
critic `k`. The same `B`, combined with the gates, sets how much each
module contributes to the *shared* representation that the actor reads:
`module_scale_m = sum_k B_{mk} g_k`, `actor_input = concat_m(z_m *
module_scale_m)`. A channel with `g_k -> 0` stops shaping the shared
representation; `B` is what makes the routing sparse/structured rather than
dense. This is the PPO analogue of the `delta_z = B e` feedback-matrix
construction in `paper/feedback_matrix_models.tex`, generalized so `B` also
carries a forward routing role (needed because PPO trains through ordinary
backprop rather than injecting `Be` as a hand-written local delta).

Channels are differentiated by **discount horizon**: `gamma_k`, `lambda_k`
start spread across `[gamma_min, gamma_max]` / `[lambda_min, lambda_max]`
(short- to long-horizon), rather than by a hand-engineered reward
decomposition -- this keeps the design agnostic to the environment's reward
structure while still giving each critic a genuinely different objective
(matching the "different aspects" motivation in `paper/report.tex`'s
multihead model, generalized from "different state views" to "different
horizons").

The actor loss uses one combined advantage, the gate-weighted mixture of
the per-channel GAE advantages: `A_t = sum_k g_k A_{k,t} / sum_k g_k`,
normalized. Each critic's regression target blends its own return with the
gate-weighted cross-channel consensus return, `alpha_k` (the
shared-vs-specific weight) controlling the mix. `B`'s columns also carry a
sparsity penalty (negative column entropy: concentrated columns are
"sparse") and an overlap penalty (squared cosine similarity between
columns), each weighted per-channel.

## Local controllers (the requested modification)

`code/paper_hpo_dqn/hypercontroller.py` is a **centralized** controller: one
process fits a coordinate-wise ridge model per hyperparameter from a single
global scalar feedback signal, jointly across all coordinates. That
requires a bottleneck with access to every coordinate's history and a joint
regression -- there is no biological analogue of a single arbiter reading
every loop's state and solving one regression to set them all.

`local_controller.py` instead gives each channel `k` its own
`LocalChannelController`, reading only signals that are local to channel
`k` or pairwise comparisons against the others that channel `k`'s own
machinery can plausibly observe:

| Statistic | What it measures | Computed from |
|---|---|---|
| `td_variance` | own signal noisiness | `Var_t(delta_{k,t})` over the window |
| `usefulness` | own critic's fit quality | explained variance of `V_k` against its own GAE return |
| `redundancy` | overlap with other channels | mean `|corr(delta_k, delta_j)|` over `j != k` |
| `gradient_conflict` | opposition to other channels in the shared trunk | mean `-cos(grad_k, grad_j)` over `j != k`, where `grad_k` is `nabla_theta_encoder [-(log pi)(a|s) A_k]` on a shared sub-batch |

Each controller keeps an EMA of these four numbers and applies bounded,
independent proportional/multiplicative update rules (no shared state, no
joint model) to produce, for its own channel only, next-window settings:
`g_k`, `B`'s column-`k` scale, critic `k`'s learning rate, `lambda_k` (and
optionally `gamma_k`), `alpha_k`, and its own sparsity/overlap pressure. Two
channels never see each other's raw stats, only the pairwise
redundancy/conflict numbers that are meaningful to compute from their own
vantage point. No regression is fit across channels or across
hyperparameters -- everything is a same-channel EMA plus a same-channel
bounded step.

This is a design choice among several defensible decentralizations (a
per-channel ridge model, as in the original HyperController, would also
qualify); the rule-based version was chosen here for transparency and low
overhead, since with `K_max` small the whole controller bank costs O(K)
scalar operations per window.

## Running

```bash
python code/ppo_local_channel_control/train.py --smoke
python code/ppo_local_channel_control/train.py \
    --env-id CartPole-v1 --total-env-steps 200000 --output-dir results/ppo_local_channel_control
```

`--smoke` runs a tiny config (3 channels, 3 modules, 640 env steps) end to
end and prints per-window diagnostics, including a `gates=[...]` line
showing the gate bank actually moving. `--output-dir` writes
`config.json`, `windows.json` (per-window PPO + controller diagnostics),
and `episode_returns.json`.

## Scope notes

- Continuous (`Box`) action spaces are supported by `ComposablePPOAgent`
  (diagonal Gaussian) but the CLI/smoke test only exercises `CartPole-v1`
  (`Discrete`).
- Single environment, no vectorized rollout workers -- fine for a smoke
  test and small classic-control runs; scaling to `HalfCheetah`-sized
  budgets would want vectorized envs like the DQN runners in
  `code/paper_hpo_dqn/`.
