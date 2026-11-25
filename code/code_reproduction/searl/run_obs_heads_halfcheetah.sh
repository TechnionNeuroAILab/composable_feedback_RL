#!/bin/bash
# SEARL-DQN with 18 independent heads (full observation + one per observation entry), --loss own.
# Discrete HalfCheetah-v4, 1M steps, pop 4, seeds 0-4 (paired with the plain SEARL baseline
# results/searl_confgate_halfcheetah_1M/B_full, seeds 0-9).
#   MODEL=A (default)  head weights fixed at 1/18 (never mutated, never trained)       -> A_fixed
#   MODEL=B            SEARL switches heads on/off (--toggle-heads); only heads that are on vote -> B_toggle
#   MODEL=C            SEARL changes a strength g_k in [0, 1] per head (--weight-heads), w = g / sum(g) -> C_weight
#   MODEL=C_STEP       as C, but a mutation moves g_k by N(0, 0.3) (clipped) instead of redrawing it     -> C_weight_step
#   MODEL=C_ADD        C from the competent end: full head at 1, the others at 0; a head mutation ADDS 0.02 to one
#                      head's g (it only gains)                                                   -> C_add_from_full
#                      after the runs: python plot_obs_heads_weights.py ../../../results/$RES/C_weight
# FRAMES (default 1M) and RES (results subfolder) can be overridden; logs are rewritten every generation.
# Out: results/searl_obs_heads_halfcheetah_1M/<arm>/seed<s>/{history.csv, heads_log.csv, population_log.csv, run.log}
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
MODEL=${MODEL:-A}
case $MODEL in
  A) ARM=A_fixed; FLAGS="" ;;
  B) ARM=B_toggle; FLAGS="--toggle-heads" ;;
  C) ARM=C_weight; FLAGS="--weight-heads" ;;
  C_STEP) ARM=C_weight_step; FLAGS="--weight-heads --weight-mut step" ;;
  C_ADD) ARM=C_add_from_full; FLAGS="--weight-heads --weight-mut add --weight-add 0.02 --init-gates full" ;;
  A_TF) ARM=A_target_full; FLAGS="--target full" ;;   # model A, TD target from the full head only
  I2) ARM=I2_inputs; FLAGS="--vote equal --evolve-inputs"; SCRIPT=searl_obs_heads_evolve.py ;;   # A + SEARL evolves head inputs
  I4R) ARM=I4_K4_remove; FLAGS="--vote equal --n-heads 4 --init-inputs all --evolve-inputs --input-mode remove"; SCRIPT=searl_obs_heads_evolve.py ;;   # 4 full-start heads, SEARL prunes inputs
  I4C) ARM=I4_K4_ctrl; FLAGS="--vote equal --n-heads 4 --init-inputs all"; SCRIPT=searl_obs_heads_evolve.py ;;   # same 4 heads, no pruning
  I4RA) ARM=I4_K4_remove_all; FLAGS="--vote equal --n-heads 4 --init-inputs all --evolve-inputs --input-mode remove --prune-full"; SCRIPT=searl_obs_heads_evolve.py ;;   # as I4R, full head prunable too
  D1) ARM=DQN1; FLAGS="--no-searl --vote equal --n-heads 1 --init-inputs all"; SCRIPT=searl_obs_heads_evolve.py ;;   # no SEARL: one full network
  D4C) ARM=DQN4_ctrl; FLAGS="--no-searl --vote equal --n-heads 4 --init-inputs all"; SCRIPT=searl_obs_heads_evolve.py ;;   # no SEARL: 4 full heads
  D4P3) ARM=DQN4_prune3; FLAGS="--no-searl --vote equal --n-heads 4 --init-inputs all --evolve-inputs --input-mode remove"; SCRIPT=searl_obs_heads_evolve.py ;;   # no SEARL: random prune of h1-h3
  D4PA) ARM=DQN4_pruneall; FLAGS="--no-searl --vote equal --n-heads 4 --init-inputs all --evolve-inputs --input-mode remove --prune-full"; SCRIPT=searl_obs_heads_evolve.py ;;   # no SEARL: random prune of all 4
  D1W) ARM=DQN1_wide; FLAGS="--no-searl --vote equal --n-heads 1 --init-inputs all --hidden 512"; SCRIPT=searl_obs_heads_evolve.py ;;   # no SEARL: one 512-unit network (~ 4 heads x 128 in weights)
  L3S) ARM=L3_split; FLAGS="--limbs split"; SCRIPT=searl_obs_heads_evolve.py; LOSS=mix ;;          # back / front / torso heads, Q summed
  L2O) ARM=L2_overlap; FLAGS="--limbs overlap"; SCRIPT=searl_obs_heads_evolve.py; LOSS=mix ;;      # back+torso / front+torso heads
  L3R) ARM=L3_random; FLAGS="--limbs random"; SCRIPT=searl_obs_heads_evolve.py; LOSS=mix ;;        # random 6/6/5 groups, same outputs
  DL3S) ARM=DQN_L3_split; FLAGS="--no-searl --limbs split"; SCRIPT=searl_obs_heads_evolve.py; LOSS=mix ;;
  DL2O) ARM=DQN_L2_overlap; FLAGS="--no-searl --limbs overlap"; SCRIPT=searl_obs_heads_evolve.py; LOSS=mix ;;
  DL3R) ARM=DQN_L3_random; FLAGS="--no-searl --limbs random"; SCRIPT=searl_obs_heads_evolve.py; LOSS=mix ;;
  *) echo "MODEL must be A, B, C, C_STEP, C_ADD, A_TF, I2, I4R, I4C, I4RA, D1, D4C, D4P3 or D4PA"; exit 1 ;;
esac
SCRIPT=${SCRIPT:-searl_obs_heads_halfcheetah.py}
FRAMES=${FRAMES:-1000000}
RES=${RES:-searl_obs_heads_halfcheetah_1M}
OUT=../../../results/$RES/$ARM; mkdir -p $OUT
SEEDS=${SEEDS:-"0 1 2 3 4"}
for s in $SEEDS; do echo $s; done | xargs -P 5 -I{} bash -c '
  o='$OUT'/seed{}; mkdir -p $o
  [ -f $o/history.csv ] || python '$SCRIPT' --seed {} --frames '$FRAMES' --loss '${LOSS:-own}' '"$FLAGS"' --out $o > $o/run.log 2>&1'
echo ALL DONE
