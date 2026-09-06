# Expected HalfCheetah reference numbers

Use these when judging reproduction success. Absolute wall-clock for HyperController will differ by GPU.

## HOOF (Paul et al.)

Table 1 HalfCheetah medians @ 5e6 steps (A2C + HOOF LR, RMSProp):

| KL ε | Median return |
|------|---------------|
| 0.01 | 1203 |
| 0.02 | 1451 |
| **0.03** | **1524** |
| 0.04 | 1325 |
| 0.05 | 1388 |
| 0.06 | 1301 |
| 0.07 | 1504 |

Table 2 (α+entropy, 1M-step budget) HalfCheetah HOOF median **702** vs grid-search expected best at sizes 1/2/5/10: −558 / −241 / 113 / 354.

Success: HOOF-A2C median within ~1 IQR of Fig 1a / Table 1 ε=0.03.

## SEARL (Franke et al.)

Table 3 HalfCheetah final actor (mean ± SE over 10 seeds):

- layers **2.8 ± 0.1**
- total nodes **1019 ± 80**
- nodes/layer **361 ± 22**

Fair protocol: SEARL should show ~10× better sample efficiency than random search / PBT on Fig 2a when x-axis counts **all** population env steps (RS curve ×20).

## HyperController (Gornet et al.)

No published scalar reward table. Match:

- Fig 1 curve ordering / shape (HC strong early on wall-clock)
- Fig 2 boxplots at t=1000
- Table I: **10/10** seeds finish for all six methods on HalfCheetah-v4

Record GPU in `machine_info.txt` when comparing wall-clock.
