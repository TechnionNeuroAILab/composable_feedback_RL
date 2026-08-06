# v1.1 Hyperparameter and Learning Diagnostics

## Executive conclusion

- **Best mean end-evaluation reward across lane counts:** `baseline100`
  (4287.4 averaged across 2/3/5/10 lanes).
- **Best mean threatened-hole avoidance:** `vis50`
  (avoidance rate 0.79 averaged across lane counts).
- These are **configuration rankings**, not isolated hyperparameter effects:
  the vis40/vis50 runs jointly changed visibility, collision reward, and final ε.
- All agents use one training seed. The 20 evaluation episodes quantify
  environment/evaluation variation, not training-seed uncertainty.

## Configurations

| Configuration | Visibility | Fall-time penalty | Collision reward | Final ε |
|---|---:|---:|---:|---:|
| `baseline100` | 25 | 100 | -10 | 0.025 |
| `pen500` | 25 | 500 | -10 | 0.025 |
| `vis40` | 40 | 100 | -100 | 0.1 |
| `vis50` | 50 | 100 | -100 | 0.1 |

## Which run is best?

| Lanes | Best end reward | Reward | Best avoidance | Avoidance rate |
|---:|---|---:|---|---:|
| 2 | `baseline100` | 5000.0 | `baseline100` | 1.00 |
| 3 | `baseline100` | 4521.8 | `pen500` | 0.86 |
| 5 | `baseline100` | 3720.8 | `vis50` | 0.58 |
| 10 | `baseline100` | 3907.2 | `vis50` | 0.83 |

Use raw reward and zero-fall success when the objective is simply to reach
5000. Use avoidance rate, near-hole lane changes, and policy concentration
when the scientific question is whether the agent learned reactive evasion.
`holes_passed` alone is not evidence of learning: it includes holes in lanes
the bike never occupied.

## Why reward plateaus lower with more lanes

1. **Falls explain the arithmetic.** The maximum is 5000 (1000 safe steps ×
   speed reward 5). A fall contributes the collision reward and also removes
   `fall_time_penalty` future rewarding steps. In the baseline, mean end reward
   drops by 1092.8 from 2 to 10 lanes while mean falls
   increase by 2.30.
2. **The observation grows while the network is unchanged.** Inputs grow from
   4 values at 2 lanes to 12 at 10 lanes, but every run uses the same
   `[256, 256]` DQN. It must learn which neighboring direction is safe from
   more lane-specific hole distances.
3. **Policy collapse is measurable.** High lane concentration, low action
   entropy, zero near-hole lane changes, or repeated outward actions at a road
   boundary indicate that the policy found a stable lane/edge strategy instead
   of reacting to visible holes.
4. **More lanes add irrelevant successes.** `passed_other_lane` grows with lane
   count even if the bike never dodges. The hole-event plot therefore separates
   avoided, collided, unprotected, and other-lane events.
5. **The 500-step penalty is too destructive for diagnosis.** One mistake can
   remove half an episode, reducing useful post-fall experience and increasing
   return variance. Its score should not be interpreted as a clean test of
   collision aversion.

## Observed policy mechanisms in these runs

- **Baseline reward is not always avoidance learning.** At 5 lanes it scores
  3720.8, but its avoidance rate
  and near-hole lane changes are both
  0.00. Its maximum
  lane occupancy is 1.00,
  evidence of a single-lane policy.
- **vis50 preserves reactive behavior at higher lane counts.** At 5 lanes its
  avoidance rate is 0.58
  with 12.5
  near-hole lane changes per episode; at 10 lanes these are
  0.83 and
  44.7.
- **vis40 shows a degenerate solution.** Lanes 2, 3, and 10 have
  the same end reward
  (2983.2) and falls (3.9),
  while their dominant lane is lane 0 and lane concentration is near 1. This
  is consistent with migrating to the first-generated lane rather than using
  the extra lanes.
- **pen500 is unstable.** Its mean last-500 reward standard deviation across
  lane counts is 980.6, versus
  484.0 for baseline100.

## How to read the figures

1. `01_learning_curves_by_lane.png`: learning speed, plateau, and instability.
2. `02_end_reward_and_success.png`: deterministic end-policy score and fraction
   of truly perfect episodes.
3. `03_hole_behavior.png`: whether reward corresponds to avoided holes rather
   than unrelated holes passing in other lanes.
4. `04_reactive_behavior.png`: policy-collapse diagnostics.
5. `05_reward_vs_avoidance.png`: separates high-scoring reactive policies from
   high-scoring passive policies.
6. `06_lane_scaling_decomposition.png`: connects extra lanes to falls, shorter
   episodes, and lower reward.
7. `07_stage_progression.png`: shows whether avoidance is acquired and retained.
8. `08_falls_and_avoidance_per_episode.png`: checkpoint learning curves for mean
   falls and avoidance rate, using the same 2×2 lane layout as figure 1.
9. `09_baseline100_training_falls_per_episode.png`: per-training-episode fall counts
   for baseline100 when `training_episodes.csv` is available.

## Statistical interpretation

- Error bars in figure 2 are percentile bootstrap 95% confidence intervals over
  the 20 deterministic evaluation episodes.
- `paired_differences.csv` compares each configuration with `baseline100` using
  matching evaluation episode IDs/seeds. A CI excluding zero is evidence of a
  consistent evaluation-seed difference for this trained seed.
- It is **not** evidence that a hyperparameter generalizes across training
  seeds. Run at least 5 independent training seeds before making a final
  hyperparameter claim.
- Training plateau metrics and deterministic end evaluation answer different
  questions. Report both; checkpoint performance can differ from the last-500
  exploratory training return.

## Recommended next experiment

Keep fall-time penalty, collision reward, and exploration fixed, and sweep only
visibility (25/40/50) across at least five training seeds. Then separately sweep
collision reward and final ε. This factorial discipline is required to identify
which individual hyperparameter causes an improvement.

## Machine-readable outputs

- `summary_table.csv`: one row per configuration × lane count.
- `paired_differences.csv`: paired end-evaluation differences from baseline.
