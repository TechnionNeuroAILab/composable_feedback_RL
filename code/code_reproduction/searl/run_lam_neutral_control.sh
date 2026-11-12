#!/bin/bash
# Control for the lambda trends: same mix arms, lam mutates/inherits as usual but is NOT used to act (--neutral-lam).
# CartPole seeds 1-6, HalfCheetah seeds 0-5, 1M frames, pop 4.   Out: results/searl_confgate_lam_neutral/<env>/<arm>/seed<s>
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate cleanrl_dqn_repro
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=$(realpath ../../../results/searl_confgate_lam_neutral)
run() { d=$OUT/$1/$2/seed$4; mkdir -p $d
  [ -f $d/history.csv ] || python searl_discrete_confgate.py --env $1 --seed $4 --pop 4 --frames 1000000 --neutral-lam --mix $3 --out $d > $d/run.log 2>&1; }
export -f run; export OUT
for s in 1 2 3 4 5 6; do for a in "M_2h --no-controller" "MG_2h " "M_obs --no-controller --heads=per_obs" "MG_obs --heads=per_obs"; do set -- $a; echo "CartPole-v1 $1 '${*:2}' $s"; done; done > $OUT.jobs 2>/dev/null || { mkdir -p $OUT; }
for s in 0 1 2 3 4 5; do for a in "M_2h --no-controller" "MG_2h " "M_obs --no-controller --heads=per_obs" "MG_obs --heads=per_obs"; do set -- $a; echo "HalfCheetah-v4 $1 '${*:2}' $s"; done; done >> $OUT.jobs
cat $OUT.jobs | xargs -P 24 -L1 bash -c 'run "$0" "$1" "$(echo $2 | tr -d "'"'"'")" $3'
echo ALL DONE > $OUT/done.log
