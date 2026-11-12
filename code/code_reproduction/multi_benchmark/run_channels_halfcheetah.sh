#!/usr/bin/env bash
# Single-agent CleanRL DQN with additive observation-group channels (cleanrl_dqn_channels.py) on discrete HalfCheetah-v4,
# 1M steps, same settings as results_halfcheetah_1M/cleanrl_dqn. Variants: additive (no loop), ac (A+C), flite (F-lite), f (F);
# append _noloop (e.g. flite_noloop) for the control with alpha_i fixed at 1.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH=/home/yaarit/miniconda3/envs/cleanrl_dqn_repro/bin:$PATH
SEEDS=${SEEDS:-0}; VARIANTS=${VARIANTS:-"additive ac flite f"}
mkdir -p logs_channels
run() { v=$1; s=$2; d=results_halfcheetah_1M/channels_$v; mkdir -p $d
  b=${v%_noloop}; x=""; [ "$b" != "$v" ] && x="--no_loop"   # <variant>_noloop = loop-off control
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python cleanrl_dqn_channels.py --variant $b $x --seed $s \
    --csv_path $d/seed$s.csv.tmp --chan_log_path $d/seed${s}_channels.csv > logs_channels/HalfCheetah-v4_${v}_seed$s.log 2>&1 \
    && mv $d/seed$s.csv.tmp $d/seed$s.csv; }
export -f run
for s in $SEEDS; do for v in $VARIANTS; do [ -f results_halfcheetah_1M/channels_$v/seed$s.csv ] || echo "$v $s"; done; done \
  | xargs -P 14 -L 1 bash -c 'run "$@"' _
echo "ALL DONE"
