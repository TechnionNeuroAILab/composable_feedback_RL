# Statistics Methodology

## Training budget
- Completed training episodes: 8000
- Configured stop criterion: 8000 episodes
- Evaluation rollouts per stage: 20 episodes

## Beginning / Middle / End snapshots
These are **three saved model checkpoints**, not time windows of one long rollout.

| Stage | When saved | Checkpoint file |
|-------|------------|-----------------|
| Beginning | Before any gradient update | `model_beginning.zip` |
| Middle | After ~50% of training (4000 episodes or timesteps) | `model_middle.zip` |
| End | After training finishes | `model_end.zip` |

Each stage is evaluated independently on **20 fresh episodes**
(seeds `10001` .. `10020`),
using the deterministic policy (`model.predict(..., deterministic=True)`).

## Per-episode plot conventions
All learning-curve x-axes use **episode index** (1, 2, 3, ...).
Loss curves aggregate logged SGD losses by **mean loss per training episode**.

Evaluation plots aggregate metrics **per evaluation episode first**, then summarize by stage:
- **Action / lane bars**: mean fraction of steps in each action/lane, averaged across the 20 episodes of that stage (error bars = std across episodes).
- **Hole outcomes**: mean count of each event type **per episode**, with std across episodes.
- **Avoidance rate**: for each eval episode, `avoided / (avoided + collisions)`, then mean ± std across episodes.
- **Near-hole lane changes**: mean count per episode, with std across episodes.
- **Falls**: mean falls per episode by stage; fall-reason counts are per-episode totals summed for display.
- **Trajectories**: episode 0 only (representative single rollout per stage).

## Hole event definitions
- **collision**: bike intersected a hole in its lane.
- **avoided**: hole was visible/threatening and passed in another lane without collision.
- **unprotected_pass**: hole was threatening but passed in the same lane without collision.
- **passed_other_lane**: hole passed while never in the bike's lane and not threatening.

## Environment settings for this run
- Lanes: 2
- Actions: 3
- Fixed speed: 5.0
- Hole mean gap (lambda): 5.0
- Fall rules: hole-collision
- Collision penalty: -10.0
- Grace steps: 5
