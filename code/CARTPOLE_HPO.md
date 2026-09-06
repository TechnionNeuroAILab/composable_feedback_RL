# CartPole learning-speed HPO

This suite optimizes the requested DQN hyperparameters:

- learning rate
- discount factor (`gamma`)
- epsilon exploration (`start`, `final`, and decay episodes)

The primary objective is normalized learning-curve AUC. A small
time-to-rolling-475 term breaks close ties. This targets *learning speed*, not
only final return.

## 1. Fast static search: Optuna + pruning

```bash
python code/cartpole_optuna_hpo.py \
  --tag optuna_cartpole_fast \
  --trials 40 --workers 4 \
  --devices cuda:0,cuda:1 \
  --episodes 1000
```

The SQLite study is resumable: rerun the same command/tag to add trials.
Outputs:

```text
paper/_tmp_b_feedb_cg/cartpole_hpo/optuna_cartpole_fast/
  study.sqlite3
  best_config.json
  summary.json
  logs/
```

For this small network, CPU workers can be competitive:

```bash
python code/cartpole_optuna_hpo.py --devices cpu --workers 8
```

## 2. Adaptive online search: Ray PB2

```bash
python code/cartpole_pb2_hpo.py \
  --tag pb2_cartpole_fast \
  --population 4 \
  --episodes 1000 \
  --episodes-per-iteration 50 \
  --gpus-per-trial 0.5
```

Resume an interrupted experiment:

```bash
python code/cartpole_pb2_hpo.py \
  --tag pb2_cartpole_fast --resume \
  --population 4 --episodes 1000
```

PB2 periodically inherits weights from strong trials and changes the five
search variables online. Its population score therefore measures the complete
adaptive procedure, not merely its final hyperparameter values.

## 3. Held-out comparison

```bash
python code/cartpole_hpo_report.py \
  --tag cartpole_hpo_comparison \
  --episodes 1000 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --optuna-tag optuna_cartpole_fast \
  --pb2-tag pb2_cartpole_fast
```

Optionally include a trained repository meta-controller:

```bash
python code/cartpole_hpo_report.py \
  --meta-checkpoint meta_controller_seed1_cartpole_ep1000_m20.pt
```

The report contains:

- learning-curve AUC
- rolling-20 episodes to return 475
- solve rate
- last-100 return
- mean ± SEM learning curves

Outputs:

```text
paper/_tmp_b_feedb_cg/cartpole_hpo/cartpole_hpo_comparison/report.json
paper/figures/conf_meta_hpo/cartpole_hpo_comparison.jpg
```

## Interpretation

The static report evaluates PB2's final hyperparameters on fresh seeds for a
configuration-only comparison. PB2's own `summary.json` reports the adaptive
population result, which also benefits from checkpoint inheritance and changing
hyperparameters. Report those two quantities separately.
