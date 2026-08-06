# v1.1 Training Run — Hyperparameters

Run label: `v1.1_plots_vis40_100penalty`  
Script: `bike_env/code/bike_dqn_multi_lanes_v1.1.py`  
Lane counts: **2, 3, 5, 10** (separate DQN runs, same settings)

## Environment

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_lanes` | 2, 3, 5, 10 | One run per lane count |
| `action_count` | 3 | Left / Stay / Right; fixed speed |
| `fixed_speed` | 5.0 | Forward speed with 3 actions |
| `visibility_range` | **40** | Holes within 40 distance units appear in obs |
| `hole_lambda` | 5.0 | Exponential gap scale (same-lane) |
| `min_same_lane_distance` | 8.0 | Base same-lane gap floor |
| `min_adjacent_hole_distance` | 20.0 | Adjacent-lane base separation |
| `adjacent_deletion_multiplier` | 2.0 | Reject lane *i* candidates within 40 of lane *i−1* |
| `fall_rules` | `hole_collision` | Falls only from hitting holes |
| `grace_steps` | 5 | No falls for first 5 steps after reset/fall |
| `episode_steps` | 1000 | Max steps per episode |
| `fall_time_penalty` | **100** | Steps removed from episode after each fall |
| `collision_penalty` | **−100** | Reward on the fall step |
| `collision_distance` | 0.8 | Hole intersection threshold |
| `reset_holes_on_fall` | false | Hole layout persists after fall |

Hole layout: lane-by-lane pre-generation; discrete lane changes (instant move to adjacent lane).

## Training (DQN)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `seed` | 1 | Same seed for all lane counts |
| `total_episodes` | 2000 | Per lane count |
| `exploration_fraction` | 0.25 | ε decays over first 25% of timesteps |
| `exploration_final_eps` | **0.1** | Minimum random-action probability |
| `learning_starts` | 5000 | No gradient updates until 5000 env steps |
| `learning_rate` | 1e-4 | SB3 DQN default in script |
| `batch_size` | 128 | |
| `policy` | MlpPolicy `[256, 256]` | |
| `eval_episodes` | 20 | Per snapshot (beginning / middle / end) |
| `checkpoint_freq` | 100000 | |
| `loss_log_freq` | 100 | |

## Outputs

- **Plots:** `bike_env/plotting/v1.1_plots_vis40_100penalty/`
- **Models / logs:** `bike_env/code/training_results_v1_1_vis40_100penalty_ep2000/`
