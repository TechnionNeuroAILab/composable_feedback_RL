# Cross-Lane Comparison Methodology

## Runs included
Matched v1 runs: 2, 3, 5, 10, 20 lanes; 3 actions; hole-only falls;
lambda=5 mean hole gap; fixed speed 5; collision penalty -10; seed 1; 8000 training episodes.

## Uncertainty
Each lane count uses **20 evaluation episodes** at the end checkpoint.
Error bars and boxplots reflect **std / distribution across those 20 episodes**, not across training seeds.

## Key metrics
- **Avoidance rate** (per eval episode): `holes_avoided / (holes_avoided + hole_collisions)`
- **Learning improvement**: end-stage mean reward minus beginning-stage mean reward
- **Near-hole lane changes**: count of steps where the agent changed lane while a visible hole was in the previous lane

## Figures
1. Overlaid 100-episode moving-average training rewards
2. End-stage scalars vs lane count (error bars = eval episode std)
3. End-stage boxplots with individual eval episodes
4. Beginning / middle / end progression by lane count
5. Hole outcome breakdown (avoided, collision, unprotected_pass)
6. Lane occupancy fractions at end checkpoint
7. Near-hole lane-change rate vs lane count
