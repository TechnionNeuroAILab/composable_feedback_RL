#!/bin/bash
# Default: CartPole-v1, population 4, 500k frames, seeds 1-10. Override with ENVID, FRAMES, SEEDS, OUT, CONFIGS, PAR.
#   A_*: aux heads drive lr (controller on)     B_*: SEARL alone (controller off, bit-identical to plain SEARL-DQN)
#   M_*: ConfGate-mix, heads also ACT (lam mutated by SEARL, init 0.2): M_* = mix only (lr gate off), MG_* = mix + lr gate;
#        *_2h = two group heads, *_obs = one head per observation entry
#   *_full: all SEARL mutations                 *_lr: lr (+ controller for A) mutation only
cd "$(dirname "$0")"
OUT=${OUT:-../../../results/searl_confgate_cartpole}
CONFIGS=${CONFIGS:-"A_full B_full A_lr B_lr"}
FRAMES=${FRAMES:-500000}
ENVID=${ENVID:-CartPole-v1}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
run() {  # name flags seed
  d=$OUT/$1/seed$3; mkdir -p $d
  [ -f $d/history.csv ] || python searl_discrete_confgate.py --seed $3 --pop 4 --env $ENVID --frames $FRAMES $2 --out $d > $d/run.log 2>&1
}
export -f run; export OUT FRAMES ENVID
for s in $SEEDS; do
  for c in $CONFIGS; do
    case $c in
      A_full) echo "A_full '' $s";; B_full) echo "B_full '--no-controller' $s";;
      A_perobs) echo "A_perobs '--heads per_obs' $s";;
      M_2h) echo "M_2h '--mix --no-controller' $s";; MG_2h) echo "MG_2h '--mix' $s";;
      M_obs) echo "M_obs '--mix --no-controller --heads per_obs' $s";; MG_obs) echo "MG_obs '--mix --heads per_obs' $s";;
      A_lr) echo "A_lr '--mutations lr_only' $s";; B_lr) echo "B_lr '--no-controller --mutations lr_only' $s";;
    esac
  done
done | xargs -P ${PAR:-40} -L1 bash -c 'run "$0" "$(echo $1 | tr -d "'"'"'")" $2'
